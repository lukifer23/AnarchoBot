import argparse
import math
import signal
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_map
import numpy as np
import yaml

from anarchobot.config import DataConfig, ModelConfig, TrainingConfig
from anarchobot.utils import cosine_lr, rotate_checkpoints
from anarchobot.tokenizer import SentencePieceTokenizer
from anarchobot_mlx.data_mlx import token_chunk_iterator, shard_paths
from anarchobot_mlx.model_mlx import TransformerLM
from anarchobot_mlx.optim_mlx import MuonAdamW
from anarchobot.training_logger import TrainingLogger, TrainingMetrics


def parse_args():
    p = argparse.ArgumentParser(description="Pre-train AnarchoBot with MLX backend.")
    p.add_argument("--config", type=Path, required=True, help="YAML config file.")
    p.add_argument("--shard-dir", type=Path, required=True, help="Directory of shards.")
    p.add_argument("--format", choices=["mlx", "npz", "txt"], default="mlx",
                   help="Shard format: pretokenized mlx/npz or raw txt.")
    return p.parse_args()


def load_configs(path: Path):
    cfg = yaml.safe_load(path.read_text())
    model_dict = dict(cfg["model"])
    float_fields = ["mlp_multiple", "dropout", "rope_theta", "norm_eps"]
    int_fields = ["vocab_size", "n_layers", "d_model", "n_heads", "max_seq_len"]
    for f in float_fields:
        if f in model_dict and isinstance(model_dict[f], str):
            model_dict[f] = float(model_dict[f])
    for f in int_fields:
        if f in model_dict and isinstance(model_dict[f], str):
            model_dict[f] = int(model_dict[f])
    model_cfg = ModelConfig(vocab_size=model_dict["vocab_size"], **{k: v for k, v in model_dict.items() if k != "vocab_size"})
    data_dict = dict(cfg["data"])
    if data_dict.get("cache_dir"):
        data_dict["cache_dir"] = Path(data_dict["cache_dir"])
    data_cfg = DataConfig(**data_dict)
    train_dict = dict(cfg["train"])
    for key in ["save_dir", "tokenizer_path", "checkpoint_path", "log_path"]:
        if train_dict.get(key):
            train_dict[key] = Path(train_dict[key])
    float_fields = ["lr", "min_lr", "weight_decay"]
    int_fields = ["warmup_steps", "ckpt_interval", "log_interval", "ckpt_keep", "total_steps", "micro_batch_size", "grad_accum_steps"]
    for f in float_fields:
        if f in train_dict and isinstance(train_dict[f], str):
            train_dict[f] = float(train_dict[f])
    for f in int_fields:
        if f in train_dict and isinstance(train_dict[f], str):
            train_dict[f] = int(train_dict[f])
    train_cfg = TrainingConfig(**train_dict)
    return model_cfg, data_cfg, train_cfg


def load_mlx_checkpoint(path: Path):
    data = np.load(path, allow_pickle=True)
    step = int(data["step"]) if "step" in data else 0
    params = {k: mx.array(v) for k, v in data.items() if k != "step"}
    return params, step


def main():
    args = parse_args()
    model_cfg, data_cfg, train_cfg = load_configs(args.config)
    if train_cfg.warmup_steps < 1:
        raise ValueError("warmup_steps must be >= 1")

    # Ensure we run on the Apple GPU (MPS) for MLX
    try:
        mx.set_default_device(mx.gpu)
    except Exception:
        pass

    tokenizer = SentencePieceTokenizer(train_cfg.tokenizer_path)
    model = TransformerLM(
        vocab_size=model_cfg.vocab_size,
        n_layers=model_cfg.n_layers,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        mlp_multiple=model_cfg.mlp_multiple,
        dropout=model_cfg.dropout,
        max_seq_len=model_cfg.max_seq_len,
        rope_theta=model_cfg.rope_theta,
        ffn_activation=model_cfg.ffn_activation,
        norm_eps=model_cfg.norm_eps,
        tie_embeddings=model_cfg.tie_embeddings,
    )
    model.train()
    optimizer = MuonAdamW(
        lr_muon=train_cfg.lr,
        lr_adam=train_cfg.lr * 1.5,
        weight_decay=train_cfg.weight_decay,
        momentum=0.95,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    start_step = 0
    if train_cfg.checkpoint_path and train_cfg.checkpoint_path.exists():
        params, start_step = load_mlx_checkpoint(train_cfg.checkpoint_path)
        model.update(params)
        print(f"Resumed from checkpoint {train_cfg.checkpoint_path} at step {start_step}")
    else:
        last_ckpt = train_cfg.save_dir / "mlx_last.npz"
        if last_ckpt.exists():
            params, start_step = load_mlx_checkpoint(last_ckpt)
            model.update(params)
            print(f"Resumed from {last_ckpt} at step {start_step}")

    seq_len = data_cfg.seq_len
    batch_size = train_cfg.micro_batch_size
    accum = train_cfg.grad_accum_steps
    total_steps = train_cfg.total_steps
    logger = TrainingLogger(
        log_dir=train_cfg.save_dir / "logs",
        experiment_name=f"mlx_pretrain_{model_cfg.d_model}d_{model_cfg.n_layers}l",
        console_log_interval=train_cfg.log_interval,
        checkpoint_log_interval=train_cfg.ckpt_interval,
    )
    logger.log_hyperparameters(
        {
            "backend": "mlx",
            "model_vocab_size": model_cfg.vocab_size,
            "model_n_layers": model_cfg.n_layers,
            "model_d_model": model_cfg.d_model,
            "model_n_heads": model_cfg.n_heads,
            "data_seq_len": data_cfg.seq_len,
            "train_total_steps": train_cfg.total_steps,
            "train_batch_size": train_cfg.micro_batch_size,
            "train_lr": train_cfg.lr,
            "train_weight_decay": train_cfg.weight_decay,
        }
    )

    print(f"Loading data from {args.shard_dir} (format: {args.format})...")
    if args.format == "txt":
        data_iter = iter(token_chunk_iterator(args.shard_dir, tokenizer, seq_len, format="txt"))
    elif args.format == "npz":
        shard_files = shard_paths(args.shard_dir, extension="npz")
        if not shard_files:
            raise ValueError(f"No npz shards found in {args.shard_dir}")
        shard_idx = 0
        shard_pos = 0
        shard_x = shard_y = None

        def load_shard(idx):
            data = np.load(shard_files[idx])
            return data["x"], data["y"]

        shard_x, shard_y = load_shard(shard_idx)

        def next_batch_npz():
            nonlocal shard_idx, shard_pos, shard_x, shard_y
            if shard_pos + batch_size > len(shard_x):
                shard_idx = (shard_idx + 1) % len(shard_files)
                shard_x, shard_y = load_shard(shard_idx)
                shard_pos = 0
            xb_np = shard_x[shard_pos : shard_pos + batch_size]
            yb_np = shard_y[shard_pos : shard_pos + batch_size]
            shard_pos += batch_size
            return mx.array(xb_np), mx.array(yb_np)

        data_iter = next_batch_npz
    elif args.format == "mlx":
        data_iter = iter(token_chunk_iterator(args.shard_dir, None, seq_len, format="mlx"))
    else:
        raise ValueError(f"Unsupported format: {args.format}")
    print("Data loaded, starting training...")

    def next_batch():
        """Get next batch - optimized for MLX."""
        if callable(data_iter):
            return data_iter()

        xs = []
        ys = []
        for _ in range(batch_size):
            try:
                x, y = next(data_iter)
                xs.append(x)
                ys.append(y)
            except StopIteration:
                print("Restarting data iterator...")
                restart = token_chunk_iterator(
                    args.shard_dir,
                    tokenizer if args.format == "txt" else None,
                    seq_len,
                    format=args.format,
                )
                x, y = next(restart)
                xs.append(x)
                ys.append(y)
        return mx.stack(xs), mx.stack(ys)

    def _tree_sum_squares(tree):
        if isinstance(tree, dict):
            return sum(_tree_sum_squares(v) for v in tree.values())
        if isinstance(tree, (list, tuple)):
            return sum(_tree_sum_squares(v) for v in tree)
        return mx.sum(mx.square(tree))

    def clip_gradients(grads, max_norm: float):
        """Global-norm clip to keep MLX path consistent with MPS training."""
        if max_norm <= 0:
            return grads, 0.0
        norm_sq = _tree_sum_squares(grads)
        norm = mx.sqrt(norm_sq + 1e-9)
        scale = mx.where(norm > max_norm, max_norm / norm, 1.0)
        return tree_map(lambda g: g * scale, grads), float(norm)

    def loss_fn(model: TransformerLM, xb, yb):
        _, loss = model(xb, yb)
        return loss

    loss_and_grad = mx.value_and_grad(loss_fn)

    tokens_processed = 0
    wall_start = time.perf_counter()
    stop_requested = {"flag": False}

    def _handle_signal(signum, frame):
        stop_requested["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    for step in range(start_step, total_steps):
        step_start = time.perf_counter()
        if stop_requested["flag"]:
            emergency = train_cfg.save_dir / "mlx_emergency.npz"
            emergency.parent.mkdir(parents=True, exist_ok=True)
            mx.savez(emergency, step=mx.array(step), **model.trainable_parameters())
            rotate_checkpoints(train_cfg.save_dir, "mlx_step_", train_cfg.ckpt_keep)
            break

        total_loss = 0.0
        grads_accum = None
        step_tokens = 0

        for _ in range(accum):
            xb, yb = next_batch()
            step_tokens += int(xb.size)
            loss, grads = loss_and_grad(model, xb, yb)
            loss = loss / accum
            grads = tree_map(lambda g: g / accum, grads)
            total_loss += float(loss)
            if grads_accum is None:
                grads_accum = grads
            else:
                grads_accum = tree_map(lambda a, b: a + b, grads_accum, grads)

        grads_accum, grad_norm = clip_gradients(grads_accum, train_cfg.max_grad_norm)

        lr = cosine_lr(step + 1, train_cfg.warmup_steps, total_steps, train_cfg.lr, train_cfg.min_lr)
        optimizer.lr_muon = lr
        optimizer.lr_adam = lr * 1.5

        params = model.trainable_parameters()
        new_params = optimizer.update(params, grads_accum)
        model.update(new_params)

        tokens_processed += step_tokens
        step_elapsed = time.perf_counter() - step_start
        elapsed = time.perf_counter() - wall_start
        tokens_per_sec = step_tokens / max(step_elapsed, 1e-6)
        tokens_per_sec_avg = tokens_processed / max(elapsed, 1e-6)
        ppl = math.exp(total_loss) if total_loss < 20 else float("inf")
        steps_done = step + 1
        steps_left = max(total_steps - steps_done, 0)
        eta_sec = steps_left * (elapsed / max(steps_done, 1))
        if step % train_cfg.log_interval == 0:
            print(
                f"[MLX] step {step} loss {total_loss:.4f} ppl {ppl:.2f} "
                f"lr {lr:.6f} grad_norm {grad_norm:.2f} "
                f"tok/s {tokens_per_sec:,.0f} avg_tok/s {tokens_per_sec_avg:,.0f} "
                f"step {step_elapsed:.2f}s elapsed {elapsed/3600:.2f}h eta {eta_sec/3600:.2f}h"
            )

        metrics = TrainingMetrics(
            step=step,
            loss=total_loss,
            learning_rate=lr,
            grad_norm=grad_norm,
            tokens_processed=step_tokens,
            throughput_tokens_per_sec=tokens_per_sec,
            step_time_sec=step_elapsed,
            timestamp=time.time(),
            perplexity=ppl,
        )
        logger.log_step(metrics)

        if step > 0 and step % train_cfg.ckpt_interval == 0:
            ckpt_path = train_cfg.save_dir / f"mlx_step_{step}.npz"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            mx.savez(ckpt_path, step=mx.array(step), **model.trainable_parameters())
            rotate_checkpoints(train_cfg.save_dir, "mlx_step_", train_cfg.ckpt_keep)

    final_ckpt = train_cfg.save_dir / "mlx_last.npz"
    mx.savez(final_ckpt, step=mx.array(total_steps), **model.trainable_parameters())
    rotate_checkpoints(train_cfg.save_dir, "mlx_step_", train_cfg.ckpt_keep)
    logger.save_final_summary()
    logger.close()


if __name__ == "__main__":
    main()
