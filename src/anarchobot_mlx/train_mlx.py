import argparse
import math
import time
from pathlib import Path
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
import numpy as np
import yaml
import psutil

from anarchobot.config import DataConfig, ModelConfig, TrainingConfig
from anarchobot.utils import cosine_lr, rotate_checkpoints
from anarchobot.tokenizer import SentencePieceTokenizer
from anarchobot_mlx.data_mlx import StreamingShardLoader, token_chunk_iterator
from anarchobot_mlx.model_mlx import TransformerLM
import pickle
from anarchobot.training_logger import TrainingLogger, TrainingMetrics


def parse_args():
    p = argparse.ArgumentParser(description="Pre-train AnarchoBot with MLX backend.")
    p.add_argument("--config", type=Path, required=True, help="YAML config file.")
    p.add_argument("--shard-dir", type=Path, required=True, help="Directory of shards.")
    p.add_argument("--format", choices=["mlx", "npz", "npy", "txt"], default="mlx",
                   help="Shard format: pretokenized mlx/npz/npy or raw txt.")
    p.add_argument("--profile-steps", type=int, default=0, help="If >0, capture per-step timings for first N steps.")
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

        def safe_mx_convert(x):
            """Convert numpy arrays back to MLX arrays, handling dtype issues"""
            if isinstance(x, np.ndarray):
                # Convert back to appropriate MLX dtype
                if x.dtype == np.float32:
                    return mx.array(x, dtype=mx.float32)
                elif x.dtype == np.float16:
                    # If saved as float16, convert to bfloat16 for MLX
                    return mx.array(x, dtype=mx.bfloat16)
                else:
                    return mx.array(x)
            return x

        params = tree_map(safe_mx_convert, payload["params"])
        opt_state = tree_map(safe_mx_convert, payload.get("opt_state", {}))
        return params, opt_state, int(payload.get("step", 0))
    # Legacy flat npz checkpoints are not compatible with the current parameter tree
    print(f"⚠️  Skipping legacy checkpoint {path} (npz format not supported for resume).")
    return None, None, 0


def main():
    # Clear Python cache to ensure latest code is loaded
    import subprocess
    import sys
    try:
        subprocess.run([sys.executable, "-m", "py_compile", __file__], check=True, capture_output=True)
    except:
        pass  # Ignore cache clearing failures

    args = parse_args()
    model_cfg, data_cfg, train_cfg = load_configs(args.config)

    # Profiling infrastructure
    profile_stats = defaultdict(list)
    profile_enabled = args.profile_steps > 0

    def profile_start(name):
        if profile_enabled:
            profile_stats[f"{name}_start"] = time.perf_counter()

    def profile_end(name):
        if profile_enabled:
            start_time = profile_stats.get(f"{name}_start")
            if start_time is not None:
                duration = time.perf_counter() - start_time
                profile_stats[name].append(duration)
                return duration
        return 0.0

    def print_profile_summary():
        if not profile_enabled or not profile_stats:
            return
        print("\n=== MLX PROFILING SUMMARY ===")
        for key, values in profile_stats.items():
            if not key.endswith("_start") and values:
                avg_time = sum(values) / len(values)
                print(f"{key}: {avg_time*1000:.2f} ms avg over {len(values)} samples")
        print("=" * 30)
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

    # Functional params/state for compiled training
    params = model.trainable_parameters()
    opt_state = None

    start_step = 0
    if train_cfg.checkpoint_path and train_cfg.checkpoint_path.exists():
        params_loaded, opt_state_loaded, start_step = load_mlx_checkpoint(train_cfg.checkpoint_path)
        if params_loaded is not None:
            params = params_loaded
            opt_state = opt_state_loaded
            model.update(params_loaded)
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
                model.update(params_loaded)
                print(f"Resumed from {last_ckpt} at step {start_step}")
            else:
                print(f"Skipped incompatible checkpoint {last_ckpt}")

    # Honor precision preference (float16/bfloat16) for performance; default to float16 on MLX
    target_dtype = None
    prec = getattr(train_cfg, "precision", "")
    if prec in ("float16", "fp16", "half", "", None):
        target_dtype = mx.float16
    elif prec in ("bfloat16", "bf16"):
        target_dtype = mx.bfloat16
    if target_dtype is not None:
        model.apply(lambda a: a.astype(target_dtype))
        params = model.trainable_parameters()
        if opt_state is not None:
            opt_state = tree_map(lambda a: a.astype(target_dtype) if isinstance(a, mx.array) else a, opt_state)

    accum = train_cfg.grad_accum_steps
    total_steps = train_cfg.total_steps
    # For profiling, surface per-step logs to see progress
    console_log_interval = 1 if args.profile_steps else train_cfg.log_interval

    logger = TrainingLogger(
        log_dir=train_cfg.save_dir / "logs",
        experiment_name=f"mlx_pretrain_{model_cfg.d_model}d_{model_cfg.n_layers}l",
        console_log_interval=console_log_interval,
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

    def loss_fn(xb, yb):
        logits, loss = model(xb, yb)
        if loss is not None:
            return loss
        return nn.losses.cross_entropy(logits, yb, reduction="mean")

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    compiled_loss_and_grad = mx.compile(loss_and_grad_fn)
    loss_scale_value = 1024.0 if target_dtype == mx.float16 else 1.0
    loss_scale = mx.array(loss_scale_value, dtype=mx.float32)

    # Flatten/unflatten helpers to keep compile input/output count manageable
    def flatten_tree(tree):
        leaves = []
        def _rec(t):
            if isinstance(t, dict):
                return ("dict", {k: _rec(v) for k, v in t.items()})
            if isinstance(t, (list, tuple)):
                subs = [_rec(v) for v in t]
                return ("list", type(t), subs)
            leaves.append(t)
            return ("leaf", len(leaves) - 1)
        spec = _rec(tree)
        return leaves, spec

    def unflatten_tree(leaves, spec):
        kind = spec[0]
        if kind == "dict":
            return {k: unflatten_tree(leaves, v) for k, v in spec[1].items()}
        if kind == "list":
            _, typ, subs = spec
            vals = [unflatten_tree(leaves, s) for s in subs]
            return typ(vals) if typ is tuple else vals
        _, idx = spec
        return leaves[idx]

    def flatten_with_spec(tree, spec):
        kind = spec[0]
        if kind == "dict":
            out = []
            for k, v in spec[1].items():
                out.extend(flatten_with_spec(tree[k], v))
            return out
        if kind == "list":
            _, typ, subs = spec
            seq = list(tree)
            out = []
            for i, s in enumerate(subs):
                out.extend(flatten_with_spec(seq[i], s))
            return out
        _, idx = spec
        return [tree]

    def init_opt_state(p_tree):
        if isinstance(p_tree, dict):
            return {k: init_opt_state(p_tree[k]) for k in p_tree}
        if isinstance(p_tree, (list, tuple)):
            return type(p_tree)(init_opt_state(p_tree[i]) for i in range(len(p_tree)))
        return {
            "exp_avg": mx.zeros_like(p_tree),
            "exp_avg_sq": mx.zeros_like(p_tree),
            "step": mx.array(0, dtype=mx.int32),
        }

    def adamw_update(p_tree, g_tree, s_tree, lr, weight_decay):
        if isinstance(p_tree, dict):
            return {k: adamw_update(p_tree[k], g_tree[k], s_tree[k], lr, weight_decay) for k in p_tree}
        if isinstance(p_tree, (list, tuple)):
            return type(p_tree)(adamw_update(p_tree[i], g_tree[i], s_tree[i], lr, weight_decay) for i in range(len(p_tree)))

        step_val = s_tree["step"] + 1
        beta1, beta2 = 0.9, 0.95
        exp_avg = s_tree["exp_avg"] * beta1 + g_tree * (1 - beta1)
        exp_avg_sq = s_tree["exp_avg_sq"] * beta2 + mx.square(g_tree) * (1 - beta2)
        bias_c1 = 1 - beta1 ** step_val
        bias_c2 = 1 - beta2 ** step_val
        denom = mx.sqrt(exp_avg_sq / bias_c2) + 1e-8
        step_update = (exp_avg / bias_c1) / denom
        new_p = p_tree * (1 - lr * weight_decay) - lr * step_update
        return {
            "param": new_p,
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
            "step": step_val,
        }

    # Initialize optimizer state and flatten both params and state
    if opt_state is None:
        opt_state = init_opt_state(params)
    params_leaves, params_spec = flatten_tree(params)
    opt_leaves, opt_spec = flatten_tree(opt_state)

    @mx.compile
    def compiled_train_step(p_leaves, s_leaves, xb, yb, lr, max_grad_norm, weight_decay):
        params_tree = unflatten_tree(p_leaves, params_spec)
        state_tree = unflatten_tree(s_leaves, opt_spec)
        model.update(params_tree)
        loss, grads = compiled_loss_and_grad(xb, yb)
        orig_loss = loss
        if loss_scale_value != 1.0:
            loss = loss / loss_scale
            grads = tree_map(lambda g: g / loss_scale, grads)
        grads, grad_norm = clip_gradients(grads, max_grad_norm)

        def _apply_update(p_t, g_t, s_t):
            if isinstance(p_t, dict):
                out_p = {}
                out_s = {}
                for k in p_t:
                    np_, ns_ = _apply_update(p_t[k], g_t[k], s_t[k])
                    out_p[k] = np_
                    out_s[k] = ns_
                return out_p, out_s
            if isinstance(p_t, (list, tuple)):
                out_p = []
                out_s = []
                for i in range(len(p_t)):
                    np_, ns_ = _apply_update(p_t[i], g_t[i], s_t[i])
                    out_p.append(np_)
                    out_s.append(ns_)
                return (type(p_t)(out_p) if isinstance(p_t, tuple) else out_p,
                        type(p_t)(out_s) if isinstance(p_t, tuple) else out_s)
            # s_t is the dict with exp_avg, exp_avg_sq, step
            step_val = s_t["step"] + 1
            beta1, beta2 = 0.9, 0.95
            exp_avg = s_t["exp_avg"] * beta1 + g_t * (1 - beta1)
            exp_avg_sq = s_t["exp_avg_sq"] * beta2 + mx.square(g_t) * (1 - beta2)
            bias_c1 = 1 - beta1 ** step_val
            bias_c2 = 1 - beta2 ** step_val
            denom = mx.sqrt(exp_avg_sq / bias_c2) + 1e-8
            step_update = (exp_avg / bias_c1) / denom
            new_p = p_t * (1 - lr * weight_decay) - lr * step_update
            new_s = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq, "step": step_val}
            return new_p, new_s

        new_params_tree, new_state_tree = _apply_update(params_tree, grads, state_tree)
        new_p_leaves = flatten_with_spec(new_params_tree, params_spec)
        new_s_leaves = flatten_with_spec(new_state_tree, opt_spec)
        return orig_loss, grad_norm, new_p_leaves, new_s_leaves

    def save_params(path: Path, step: int):
        def safe_numpy_convert(x):
            """Convert MLX arrays to numpy, handling different dtypes properly"""
            if isinstance(x, mx.array):
                # Handle bfloat16 conversion - MLX bfloat16 might not convert directly to numpy
                if x.dtype == mx.bfloat16:
                    # Convert to float32 first, then to numpy
                    return np.array(x.astype(mx.float32))
                else:
                    return np.array(x)
            return x

        current_params = model.trainable_parameters()
        current_opt = unflatten_tree(opt_leaves, opt_spec)
        payload = {
            "step": step,
            "params": tree_map(safe_numpy_convert, current_params),
            "opt_state": tree_map(safe_numpy_convert, current_opt),
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
        warmup_x = []
        warmup_y = []
        for _ in range(accum):
            xb_w, yb_w = next_batch()
            warmup_x.append(xb_w)
            warmup_y.append(yb_w)
        xb_w = mx.concatenate(warmup_x, axis=0) if len(warmup_x) > 1 else warmup_x[0]
        yb_w = mx.concatenate(warmup_y, axis=0) if len(warmup_y) > 1 else warmup_y[0]
        log_batch_stats(xb_w, yb_w, "warmup")
        loss_w, grad_norm_w, params_leaves, opt_leaves = compiled_train_step(
            params_leaves, opt_leaves, xb_w, yb_w, train_cfg.lr, train_cfg.max_grad_norm, train_cfg.weight_decay
        )
        model.update(unflatten_tree(params_leaves, params_spec))
    except Exception as e:
        raise RuntimeError(f"Warmup failed: {e}")
    print("Warmup complete, starting training loop...")

    tokens_processed = 0
    wall_start = time.perf_counter()
    step = start_step  # Initialize step for exception handling

    try:
        for step in range(start_step, total_steps):
            step_start = time.perf_counter()
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
            profile_start("data_prep")
            xb_step = mx.concatenate(xb_list, axis=0) if len(xb_list) > 1 else xb_list[0]
            yb_step = mx.concatenate(yb_list, axis=0) if len(yb_list) > 1 else yb_list[0]
            profile_end("data_prep")

            lr = cosine_lr(step + 1, train_cfg.warmup_steps, total_steps, train_cfg.lr, train_cfg.min_lr)

            profile_start("training_step")
            loss, grad_norm, params_leaves, opt_leaves = compiled_train_step(
                params_leaves, opt_leaves, xb_step, yb_step, lr, train_cfg.max_grad_norm, train_cfg.weight_decay
            )
            model.update(unflatten_tree(params_leaves, params_spec))
            profile_end("training_step")

            # Validate training step results - no silent failures
            loss_val = float(loss)
            grad_norm_val = float(grad_norm)

            if math.isnan(loss_val) or math.isinf(loss_val):
                raise RuntimeError(f"Invalid loss detected: {loss_val}")
            if math.isnan(grad_norm_val) or math.isinf(grad_norm_val):
                raise RuntimeError(f"Invalid grad_norm detected: {grad_norm_val}")
            if not (0 <= loss_val <= 100):  # Reasonable bounds for cross-entropy loss
                raise RuntimeError(f"Suspicious loss value: {loss_val}")

            total_loss = loss_val

            if args.profile_steps and step < args.profile_steps:
                data_time = profile_stats.get("data_prep", [0])[-1] * 1000
                step_time = profile_stats.get("training_step", [0])[-1] * 1000
                print(f"[PROFILE] step {step} data_prep {data_time:.2f}ms training_step {step_time:.2f}ms total {(time.perf_counter()-step_t0)*1000:.2f}ms")
                print(f"[DEBUG] Step {step} completed, preparing next batch...")

            tokens_processed += step_tokens
            step_elapsed = time.perf_counter() - step_start
            elapsed = time.perf_counter() - wall_start
            tokens_per_sec = step_tokens / max(step_elapsed, 1e-6)
            tokens_per_sec_avg = tokens_processed / max(elapsed, 1e-6)
            ppl = math.exp(total_loss) if total_loss < 20 else float("inf")
            steps_done = step + 1
            steps_left = max(total_steps - steps_done, 0)
            eta_sec = steps_left * (elapsed / max(steps_done, 1))
            if step % console_log_interval == 0:
                # Rough memory estimate from param count (params + grads + opt state)
                def _count_elems(tree):
                    if isinstance(tree, dict):
                        return sum(_count_elems(v) for v in tree.values())
                    if isinstance(tree, (list, tuple)):
                        return sum(_count_elems(v) for v in tree)
                    if isinstance(tree, mx.array):
                        return int(tree.size)
                    return 0
                param_elems = _count_elems(model.trainable_parameters())
                estimated_mb = int(param_elems * 4 * 3 / 1024 / 1024)  # assume 4 bytes, params+grads+opt
                print(
                    f"[MLX] step {step} loss {total_loss:.4f} ppl {ppl:.2f} "
                    f"lr {lr:.6f} grad_norm {grad_norm:.2f} "
                    f"tok/s {tokens_per_sec:,.0f} avg_tok/s {tokens_per_sec_avg:,.0f} "
                    f"step {step_elapsed:.2f}s elapsed {elapsed/3600:.2f}h eta {eta_sec/3600:.2f}h mem ~{estimated_mb}MB"
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
            save_params(ckpt_path, step)
            rotate_checkpoints(train_cfg.save_dir, "mlx_step_", train_cfg.ckpt_keep)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, saving emergency checkpoint...")
        emergency = train_cfg.save_dir / "mlx_emergency.pkl"
        save_params(emergency, step if 'step' in locals() else 0)
        rotate_checkpoints(train_cfg.save_dir, "mlx_step_", train_cfg.ckpt_keep)

    final_ckpt = train_cfg.save_dir / "mlx_last.pkl"
    save_params(final_ckpt, total_steps)
    rotate_checkpoints(train_cfg.save_dir, "mlx_step_", train_cfg.ckpt_keep)
    # Print profiling summary
    print_profile_summary()

    # Only save final summary if we actually ran some training steps
    if logger.metrics_history:
        logger.save_final_summary()
    else:
        print("📈 Training already complete - no new steps to summarize")

    logger.close()


if __name__ == "__main__":
    main()
