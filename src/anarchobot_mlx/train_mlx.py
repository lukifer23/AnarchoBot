import argparse
import math
import signal
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
import yaml

from anarchobot.config import DataConfig, ModelConfig, TrainingConfig
from anarchobot.utils import rotate_checkpoints
from anarchobot.tokenizer import SentencePieceTokenizer
from anarchobot_mlx.data_mlx import token_chunk_iterator
from anarchobot_mlx.model_mlx import TransformerLM
from anarchobot_mlx.optim_mlx import MuonAdamW


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


def main():
    args = parse_args()
    model_cfg, data_cfg, train_cfg = load_configs(args.config)
    if train_cfg.warmup_steps < 1:
        raise ValueError("warmup_steps must be >= 1")

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
    optimizer = MuonAdamW(
        lr_muon=train_cfg.lr,
        lr_adam=train_cfg.lr * 1.5,
        weight_decay=train_cfg.weight_decay,
        momentum=0.95,
    )

    seq_len = data_cfg.seq_len
    batch_size = train_cfg.micro_batch_size
    accum = train_cfg.grad_accum_steps
    total_steps = train_cfg.total_steps

    print(f"Loading data from {args.shard_dir} (format: {args.format})...")
    if args.format == "txt":
        data_iter = token_chunk_iterator(args.shard_dir, tokenizer, seq_len, format="txt")
    elif args.format in ("mlx", "npz"):
        data_iter = token_chunk_iterator(args.shard_dir, None, seq_len, format=args.format)
    else:
        raise ValueError(f"Unsupported format: {args.format}")
    print("Data loaded, starting training...")

    def next_batch():
        """Get next batch - optimized for MLX."""
        xs = []
        ys = []
        for _ in range(batch_size):
            try:
                x, y = next(data_iter)
                xs.append(x)
                ys.append(y)
            except StopIteration:
                # Restart data iterator when exhausted
                print("Restarting data iterator...")
                new_iter = token_chunk_iterator(
                    args.shard_dir,
                    tokenizer if args.format == "txt" else None,
                    seq_len,
                    format=args.format,
                )
                x, y = next(new_iter)
                xs.append(x)
                ys.append(y)
        return mx.stack(xs), mx.stack(ys)

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

    for step in range(total_steps):
        if stop_requested["flag"]:
            emergency = train_cfg.save_dir / "mlx_emergency.npz"
            emergency.parent.mkdir(parents=True, exist_ok=True)
            mx.savez(emergency, **model.trainable_parameters())
            rotate_checkpoints(train_cfg.save_dir, "mlx_step_", train_cfg.ckpt_keep)
            break

        total_loss = 0.0
        grads_accum = None
        step_tokens = 0

        for _ in range(accum):
            xb, yb = next_batch()
            step_tokens += int(xb.size)
            loss, grads = loss_and_grad(model, xb, yb)
            total_loss += float(loss)
            if grads_accum is None:
                grads_accum = grads
            else:
                grads_accum = tree_map(lambda a, b: a + b, grads_accum, grads)

        grads_accum = tree_map(lambda g: g / accum, grads_accum)
        if train_cfg.max_grad_norm > 0:
            grads_accum = tree_map(
                lambda g: mx.clip(g, -train_cfg.max_grad_norm, train_cfg.max_grad_norm),
                grads_accum,
            )

        new_params = optimizer.update(model.trainable_parameters(), grads_accum)
        model.update(new_params)

        tokens_processed += step_tokens
        elapsed = time.perf_counter() - wall_start
        tokens_per_sec = tokens_processed / max(elapsed, 1e-6)
        ppl = math.exp(total_loss) if total_loss < 20 else float("inf")
        steps_done = step + 1
        steps_left = max(total_steps - steps_done, 0)
        eta_sec = steps_left * (elapsed / max(steps_done, 1))
        if step % train_cfg.log_interval == 0:
            print(
                f"[MLX] step {step} loss {total_loss:.4f} ppl {ppl:.2f} tok/s {tokens_per_sec:,.0f} "
                f"elapsed {elapsed/3600:.2f}h eta {eta_sec/3600:.2f}h"
            )

        if step > 0 and step % train_cfg.ckpt_interval == 0:
            ckpt_path = train_cfg.save_dir / f"mlx_step_{step}.npz"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            mx.savez(ckpt_path, **model.trainable_parameters())
            rotate_checkpoints(train_cfg.save_dir, "mlx_step_", train_cfg.ckpt_keep)

    final_ckpt = train_cfg.save_dir / "mlx_last.npz"
    mx.savez(final_ckpt, **model.trainable_parameters())
    rotate_checkpoints(train_cfg.save_dir, "mlx_step_", train_cfg.ckpt_keep)


if __name__ == "__main__":
    main()
