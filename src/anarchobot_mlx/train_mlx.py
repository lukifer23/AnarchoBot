import argparse
import math
import signal
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_flatten
import numpy as np
import yaml

from anarchobot.config import DataConfig, ModelConfig, TrainingConfig
from anarchobot.utils import cosine_lr, rotate_checkpoints
from anarchobot.tokenizer import SentencePieceTokenizer
from anarchobot_mlx.data_mlx import StreamingShardLoader
from anarchobot_mlx.model_mlx import TransformerLM
from anarchobot_mlx.optim_mlx import zeropower_via_newtonschulz5
import pickle
from anarchobot.training_logger import TrainingLogger, TrainingMetrics


def parse_args():
    p = argparse.ArgumentParser(description="Pre-train AnarchoBot with MLX backend.")
    p.add_argument("--config", type=Path, required=True, help="YAML config file.")
    p.add_argument("--shard-dir", type=Path, required=True, help="Directory of shards.")
    p.add_argument("--format", choices=["mlx", "npz", "npy", "txt"], default="mlx",
                   help="Shard format: pretokenized mlx/npz/npy or raw txt.")
    p.add_argument("--profile-steps", type=int, default=0, help="If >0, capture per-step timings for first N steps.")
    p.add_argument("--muon-ns-steps", type=int, default=1, help="Newton-Schulz steps for Muon (lower for speed).")
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
    if data_dict.get("shard_dir"):
        data_dict["shard_dir"] = Path(data_dict["shard_dir"])
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
    if path.suffix == ".pkl":
        payload = pickle.loads(path.read_bytes())
        params = tree_map(lambda a: mx.array(a) if isinstance(a, np.ndarray) else a, payload["params"])
        opt_state = tree_map(lambda a: mx.array(a) if isinstance(a, np.ndarray) else a, payload.get("opt_state", {}))
        return params, opt_state, int(payload.get("step", 0))
    # Legacy flat npz checkpoints are not compatible with the current parameter tree
    print(f"⚠️  Skipping legacy checkpoint {path} (npz format not supported for resume).")
    return None, None, 0


def main():
    args = parse_args()
    model_cfg, data_cfg, train_cfg = load_configs(args.config)
    if train_cfg.warmup_steps < 1:
        raise ValueError("warmup_steps must be >= 1")

    # Force MLX GPU; fail fast otherwise.
    try:
        mx.set_default_device(mx.gpu)
    except Exception as e:
        raise RuntimeError(f"Unable to set MLX default device to GPU: {e}")
    dev = mx.default_device()
    # Check if device is GPU - MLX uses DeviceType enum
    if dev.type != mx.DeviceType.gpu:
        raise RuntimeError(
            f"MLX default device is {dev}, expected GPU. "
            "Ensure MLX is installed with Metal support and that a Metal GPU is available."
        )
    print(f"Using MLX device: {dev}")

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
    seq_len = data_cfg.seq_len
    batch_size = train_cfg.micro_batch_size

    # MLX doesn't support gradient checkpointing like PyTorch yet
    # model.enable_gradient_checkpointing(train_cfg.gradient_checkpointing)
    model.train()

    # Functional params/state for training
    params = model.trainable_parameters()
    opt_state = None
    start_step = 0
    if train_cfg.checkpoint_path and train_cfg.checkpoint_path.exists():
        params_loaded, opt_state_loaded, start_step = load_mlx_checkpoint(train_cfg.checkpoint_path)
        if params_loaded is not None:
            params = params_loaded
            opt_state = opt_state_loaded
            model.update(params)
            print(f"Resumed from checkpoint {train_cfg.checkpoint_path} at step {start_step}")
        else:
            print(f"Skipped incompatible checkpoint {train_cfg.checkpoint_path}")
    else:
        last_ckpt = train_cfg.save_dir / "mlx_last.pkl"
        if last_ckpt.exists():
            params_loaded, opt_state_loaded, start_step = load_mlx_checkpoint(last_ckpt)
            if params_loaded is not None:
                params = params_loaded
                opt_state = opt_state_loaded
                model.update(params)
                print(f"Resumed from {last_ckpt} at step {start_step}")
            else:
                print(f"Skipped incompatible checkpoint {last_ckpt}")

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
    elif args.format in ("npz", "mlx", "npy"):
        # Use optimized streaming loader for better performance
        data_loader = StreamingShardLoader(args.shard_dir, batch_size, seq_len, args.format)
        data_iter = iter(data_loader)
    else:
        raise ValueError(f"Unsupported format: {args.format}")

    def next_batch():
        """Get next batch - optimized for MLX."""
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            print("Restarting data iterator...")
            if args.format == "txt":
                data_iter = iter(token_chunk_iterator(args.shard_dir, tokenizer, seq_len, format="txt"))
            else:
                data_loader = StreamingShardLoader(args.shard_dir, batch_size, seq_len, args.format)
                data_iter = iter(data_loader)
            return next(data_iter)

    def _tree_sum_squares(tree):
        if isinstance(tree, dict):
            return sum(_tree_sum_squares(v) for v in tree.values())
        if isinstance(tree, (list, tuple)):
            return sum(_tree_sum_squares(v) for v in tree)
        return mx.sum(mx.square(tree))

    def clip_gradients(grads, max_norm: float):
        """Global-norm clip to keep MLX path consistent with MPS training."""
        if max_norm <= 0:
            return grads, mx.array(0.0)
        norm_sq = _tree_sum_squares(grads)
        norm = mx.sqrt(norm_sq + 1e-9)
        scale = mx.where(norm > max_norm, max_norm / norm, 1.0)
        return tree_map(lambda g: g * scale, grads), norm

    def loss_fn(p, xb, yb):
        logits, loss = model.apply(p, xb, targets=yb)
        if loss is None:
            loss = nn.losses.cross_entropy(logits, yb, reduction="mean")
        return loss

    loss_and_grad_fn = mx.value_and_grad(loss_fn, argnums=0)

    def init_opt_state(p_tree, prefix=""):
        if isinstance(p_tree, dict):
            return {k: init_opt_state(p_tree[k], prefix + k + ".") for k in p_tree}
        if isinstance(p_tree, (list, tuple)):
            return type(p_tree)(init_opt_state(p_tree[i], prefix + f"{i}.") for i in range(len(p_tree)))
        name = prefix[:-1] if prefix.endswith(".") else prefix
        if p_tree.ndim >= 2 and "embed" not in name and "lm_head" not in name:
            return {"momentum": mx.zeros_like(p_tree)}
        else:
            return {
                "exp_avg": mx.zeros_like(p_tree),
                "exp_avg_sq": mx.zeros_like(p_tree),
                "step": mx.array(0, dtype=mx.int32),
            }

    def muon_adam_update(p_tree, g_tree, s_tree, lr_muon, lr_adam, momentum, ns_steps, prefix=""):
        if isinstance(p_tree, dict):
            out_p = {}
            out_s = {}
            for k in p_tree:
                child_state = s_tree.get(k) if isinstance(s_tree, dict) else None
                if child_state is None:
                    child_state = init_opt_state(p_tree[k], prefix + k + ".")
                np_, ns_ = muon_adam_update(p_tree[k], g_tree[k], child_state, lr_muon, lr_adam, momentum, ns_steps, prefix + k + ".")
                out_p[k] = np_
                out_s[k] = ns_
            return out_p, out_s
        if isinstance(p_tree, (list, tuple)):
            out_p = []
            out_s = []
            for i in range(len(p_tree)):
                child_state = s_tree[i] if isinstance(s_tree, (list, tuple)) else None
                if child_state is None:
                    child_state = init_opt_state(p_tree[i], prefix + f"{i}.")
                np_, ns_ = muon_adam_update(p_tree[i], g_tree[i], child_state, lr_muon, lr_adam, momentum, ns_steps, prefix + f"{i}.")
                out_p.append(np_)
                out_s.append(ns_)
            return type(p_tree)(out_p), type(p_tree)(out_s)

        name = prefix[:-1] if prefix.endswith(".") else prefix
        if s_tree is None:
            s_tree = init_opt_state(p_tree, prefix)
        if p_tree.ndim >= 2 and "embed" not in name and "lm_head" not in name:
            mom = s_tree["momentum"] * momentum + g_tree * (1 - momentum)
            update = zeropower_via_newtonschulz5(mom, steps=ns_steps) if ns_steps > 0 else mom
            new_p = p_tree * (1 - lr_muon * train_cfg.weight_decay) - lr_muon * update
            return new_p, {"momentum": mom}
        else:
            step_val = s_tree["step"] + 1
            beta1, beta2 = 0.9, 0.95
            exp_avg = s_tree["exp_avg"] * beta1 + g_tree * (1 - beta1)
            exp_avg_sq = s_tree["exp_avg_sq"] * beta2 + mx.square(g_tree) * (1 - beta2)
            bias_c1 = 1 - beta1 ** step_val
            bias_c2 = 1 - beta2 ** step_val
            denom = mx.sqrt(exp_avg_sq / bias_c2) + 1e-8
            step_update = (exp_avg / bias_c1) / denom
            new_p = p_tree * (1 - lr_adam * train_cfg.weight_decay) - lr_adam * step_update
            return new_p, {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq, "step": step_val}

    def eval_tree(tree):
        res = tree_flatten(tree)
        if isinstance(res, tuple):
            leaves = res[0]
        else:
            leaves = res
        if not isinstance(leaves, (list, tuple)):
            leaves = [leaves]
        leaves = [leaf for leaf in leaves if leaf is not None]
        if leaves:
            mx.eval(*leaves)

    def flatten_arrays(tree, prefix=""):
        """Flatten nested params/grads to a flat dict of arrays for robust saving."""
        flat = {}
        if isinstance(tree, dict):
            for k, v in tree.items():
                flat.update(flatten_arrays(v, prefix + k + "."))
        elif isinstance(tree, (list, tuple)):
            for i, v in enumerate(tree):
                flat.update(flatten_arrays(v, prefix + f"{i}."))
        else:
            if isinstance(tree, mx.array):
                name = prefix[:-1] if prefix.endswith(".") else prefix
                flat[name] = tree
        return flat

    def save_params(path: Path, params, opt_state, step: int):
        payload = {
            "step": step,
            "params": tree_map(lambda a: np.array(a) if isinstance(a, mx.array) else a, params),
            "opt_state": tree_map(lambda a: np.array(a) if isinstance(a, mx.array) else a, opt_state),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(payload))

    def log_batch_stats(xb, yb, prefix="batch"):
        try:
            print(f"{prefix}: x shape {xb.shape}, y shape {yb.shape}, dtype {xb.dtype}")
        except Exception:
            pass

    print("Warmup: running first MLX step to initialize kernels (can take ~20-30s)...")
    try:
        xb_w, yb_w = next_batch()
        log_batch_stats(xb_w, yb_w, "warmup")
        if opt_state is None:
            opt_state = init_opt_state(params)
        loss_w, grads_w = loss_and_grad_fn(params, xb_w, yb_w)
        grads_w, _ = clip_gradients(grads_w, train_cfg.max_grad_norm)
        params, opt_state = muon_adam_update(
            params,
            grads_w,
            opt_state,
            lr_muon=train_cfg.lr,
            lr_adam=train_cfg.lr * 1.5,
            momentum=0.95,
            ns_steps=args.muon_ns_steps,
        )
    except Exception as e:
        raise RuntimeError(f"Warmup failed: {e}")
    print("Warmup complete, starting training loop...")

    tokens_processed = 0
    wall_start = time.perf_counter()
    stop_requested = {"flag": False}
    step = start_step  # Initialize step for exception handling

    def _handle_signal(signum, frame):
        stop_requested["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        for step in range(start_step, total_steps):
            step_start = time.perf_counter()
            if stop_requested["flag"]:
                emergency = train_cfg.save_dir / "mlx_emergency.pkl"
                save_params(emergency, model.trainable_parameters(), step)
                rotate_checkpoints(train_cfg.save_dir, "mlx_step_", train_cfg.ckpt_keep)
                break

            total_loss = 0.0
            step_tokens = 0

            xb_list = []
            yb_list = []
            for micro in range(accum):
                xb, yb = next_batch()
                step_tokens += int(xb.size)
                xb_list.append(xb)
                yb_list.append(yb)

            step_t0 = time.perf_counter()
            xb_step = mx.concatenate(xb_list, axis=0) if len(xb_list) > 1 else xb_list[0]
            yb_step = mx.concatenate(yb_list, axis=0) if len(yb_list) > 1 else yb_list[0]
            lr = cosine_lr(step + 1, train_cfg.warmup_steps, total_steps, train_cfg.lr, train_cfg.min_lr)
            loss, grads = loss_and_grad_fn(params, xb_step, yb_step)
            grads, grad_norm = clip_gradients(grads, train_cfg.max_grad_norm)
            params, opt_state = muon_adam_update(params, grads, opt_state, lr_muon=lr, lr_adam=lr * 1.5, momentum=0.95, ns_steps=args.muon_ns_steps)

            total_loss = float(loss)
            grad_norm_val = float(grad_norm)

            if args.profile_steps and step < args.profile_steps:
                print(f"[PROFILE] step {step} training_step {(time.perf_counter()-step_t0)*1000:.2f} ms")

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
                grad_norm=grad_norm_val,
                tokens_processed=step_tokens,
                throughput_tokens_per_sec=tokens_per_sec,
                step_time_sec=step_elapsed,
                timestamp=time.time(),
                perplexity=ppl,
            )
            logger.log_step(metrics)

        if step > 0 and step % train_cfg.ckpt_interval == 0:
            ckpt_path = train_cfg.save_dir / f"mlx_step_{step}.pkl"
            save_params(ckpt_path, params, opt_state, step)
            rotate_checkpoints(train_cfg.save_dir, "mlx_step_", train_cfg.ckpt_keep)
    except KeyboardInterrupt:
        stop_requested["flag"] = True
        print("KeyboardInterrupt received, saving emergency checkpoint...")
        emergency = train_cfg.save_dir / "mlx_emergency.pkl"
        save_params(emergency, params, opt_state, step if 'step' in locals() else 0)
        rotate_checkpoints(train_cfg.save_dir, "mlx_step_", train_cfg.ckpt_keep)

    final_ckpt = train_cfg.save_dir / "mlx_last.pkl"
    save_params(final_ckpt, params, opt_state, total_steps)
    rotate_checkpoints(train_cfg.save_dir, "mlx_step_", train_cfg.ckpt_keep)
    # Only save final summary if we actually ran some training steps
    if logger.metrics_history:
        logger.save_final_summary()
    else:
        print("📈 Training already complete - no new steps to summarize")

    logger.close()


if __name__ == "__main__":
    main()
