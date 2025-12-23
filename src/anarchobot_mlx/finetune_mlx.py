#!/usr/bin/env python
"""
Supervised fine-tuning for AnarchoBot with MLX backend.
Requires pre-tokenized shards (npy/npz/legacy mlx) generated via scripts/prepare_sft_shards.py.
"""
import argparse
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten

from anarchobot.config import load_yaml_config
from anarchobot.training_logger import TrainingLogger, TrainingMetrics
from anarchobot.utils import cosine_lr, rotate_checkpoints
from anarchobot_mlx.checkpointing import load_checkpoint, save_checkpoint
from anarchobot_mlx.data_mlx import StreamingShardLoader
from anarchobot_mlx.model_mlx import TransformerLM


def parse_args():
    p = argparse.ArgumentParser(description="Supervised fine-tuning for AnarchoBot with MLX.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True, help="Pretrained checkpoint (.pkl).")
    p.add_argument("--resume", type=Path, help="Resume from existing SFT checkpoint (.pkl).")
    p.add_argument("--shard-dir", type=Path, help="Override data.shard_dir for pretokenized shards.")
    p.add_argument("--format", choices=["npy", "npz", "mlx"], help="Override shard format.")
    return p.parse_args()


def _target_dtype(precision: str):
    prec = precision.lower()
    if prec in ("float16", "fp16", "half"):
        return mx.float16
    if prec in ("bfloat16", "bf16"):
        return mx.bfloat16
    return None


def _cast_model_precision(model: nn.Module, precision: str):
    target = _target_dtype(precision)
    if target is None:
        return

    def cast_fn(x):
        if isinstance(x, mx.array) and mx.issubdtype(x.dtype, mx.floating):
            return x.astype(target)
        return x

    params = tree_map(cast_fn, model.parameters())
    model.update(params)


def _clip_gradients(grads, max_norm: float) -> tuple:
    if not max_norm or max_norm <= 0:
        return grads, 0.0
    flat = [g for g in tree_flatten(grads)[0] if hasattr(g, "dtype")]
    if not flat:
        return grads, 0.0

    total_sq = None
    for g in flat:
        g32 = g.astype(mx.float32)
        val = mx.sum(g32 * g32)
        total_sq = val if total_sq is None else total_sq + val
    total_norm = mx.sqrt(total_sq) if total_sq is not None else mx.array(0.0, dtype=mx.float32)
    norm_scalar = float(total_norm)
    if norm_scalar > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        grads = tree_map(lambda g: g * scale if hasattr(g, "dtype") else g, grads)
    return grads, norm_scalar


def main():
    # Clear Python cache
    import subprocess, sys
    try: subprocess.run([sys.executable, "-m", "py_compile", __file__], check=True, capture_output=True)
    except: pass

    args = parse_args()
    model_cfg, data_cfg, train_cfg = load_yaml_config(args.config)
    train_cfg.save_dir.mkdir(parents=True, exist_ok=True)

    mx.set_default_device(mx.gpu)
    print(f"Using MLX device: {mx.default_device()}")

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
    precision = getattr(train_cfg, "precision", "float16")
    _cast_model_precision(model, precision)

    if getattr(train_cfg, "optimizer", "adamw") != "adamw":
        print("Muon is not available on MLX; using AdamW for SFT.")
    optimizer = optim.AdamW(learning_rate=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, target_precision=precision)
        print(f"Resumed SFT from {args.resume} at step {start_step}")
    else:
        start_step = 0
        load_checkpoint(args.base_checkpoint, model, optimizer=None, load_optimizer=False, target_precision=precision)
        print(f"Loaded base checkpoint {args.base_checkpoint}")

    shard_dir = args.shard_dir or data_cfg.shard_dir
    if shard_dir is None:
        raise ValueError(
            "SFT with MLX requires pretokenized shards. "
            "Set data.shard_dir in config or pass --shard-dir after running scripts/prepare_sft_shards.py."
        )
    shard_format = args.format or getattr(data_cfg, "format", None) or "npy"

    seq_len = data_cfg.seq_len
    batch_size = train_cfg.micro_batch_size
    data_loader = StreamingShardLoader(
        shard_dir, batch_size, seq_len, shard_format, prefetch_batches=2, shuffle=True
    )
    data_iter = iter(data_loader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_loader = StreamingShardLoader(
                shard_dir, batch_size, seq_len, shard_format, prefetch_batches=2, shuffle=True
            )
            data_iter = iter(data_loader)
            return next(data_iter)

    def loss_fn(model, x, y):
        logits, _ = model(x)
        logits = logits.astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        return loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    def accumulate_step(accum: int):
        total_loss = 0.0
        tokens = 0
        grad_accum = None
        for _ in range(accum):
            xb, yb = next_batch()
            tokens += int(xb.size)
            loss, grads = loss_and_grad_fn(model, xb, yb)
            total_loss += loss.item()
            grad_accum = grads if grad_accum is None else tree_map(lambda a, b: a + b, grad_accum, grads)
        grad_accum = tree_map(lambda g: g / accum, grad_accum)
        grad_accum, grad_norm = _clip_gradients(grad_accum, getattr(train_cfg, "max_grad_norm", 0.0))
        optimizer.update(model, grad_accum)
        mx.eval(model.parameters(), optimizer.state)
        return total_loss / accum, tokens, grad_norm

    if getattr(train_cfg, "compile", False):
        accumulate_step = mx.compile(accumulate_step)

    logger = TrainingLogger(
        log_dir=train_cfg.save_dir / "logs",
        experiment_name=f"mlx_sft_{model_cfg.d_model}d_{model_cfg.n_layers}l",
        console_log_interval=train_cfg.log_interval,
        checkpoint_log_interval=train_cfg.ckpt_interval,
    )
    tokens_per_step = train_cfg.tokens_per_step(seq_len)
    logger.log_hyperparameters(
        {
            **train_cfg.as_logging_dict(),
            "tokens_per_step": tokens_per_step,
            "total_tokens": train_cfg.total_tokens(seq_len),
            "seq_len": seq_len,
            "shard_format": shard_format,
        }
    )

    print("Starting SFT training loop...")
    accum = max(1, train_cfg.grad_accum_steps)
    wall_start = time.perf_counter()
    tokens_processed = 0
    last_step = start_step - 1

    xb, yb = next_batch()
    warm_loss, _ = loss_and_grad_fn(model, xb, yb)
    mx.eval(warm_loss)
    print(f"Warmup forward pass loss: {warm_loss.item():.4f}")

    try:
        for step in range(start_step, train_cfg.total_steps):
            step_start = time.perf_counter()
            lr = cosine_lr(step + 1, train_cfg.warmup_steps, train_cfg.total_steps, train_cfg.lr, train_cfg.min_lr)
            optimizer.learning_rate = lr

            loss_val, step_tokens, grad_norm = accumulate_step(accum)

            step_time = time.perf_counter() - step_start
            tokens_processed += step_tokens
            last_step = step

            tps = step_tokens / max(step_time, 1e-6)
            tps_avg = tokens_processed / max(time.perf_counter() - wall_start, 1e-6)

            if step % train_cfg.log_interval == 0:
                print(f"Step {step}: loss {loss_val:.4f}, lr {lr:.6f}, tok/s {tps:.0f} (avg {tps_avg:.0f})")

            logger.log_step(
                TrainingMetrics(
                    step=step,
                    loss=loss_val,
                    learning_rate=lr,
                    grad_norm=grad_norm,
                    tokens_processed=step_tokens,
                    throughput_tokens_per_sec=tps,
                    step_time_sec=step_time,
                    timestamp=time.time(),
                    perplexity=math.exp(loss_val) if loss_val < 20 else float("inf"),
                )
            )

            if step > 0 and step % train_cfg.ckpt_interval == 0:
                ckpt_path = train_cfg.save_dir / f"sft_step_{step}.pkl"
                save_checkpoint(ckpt_path, step, model, optimizer)
                rotate_checkpoints(train_cfg.save_dir, "sft_step_", train_cfg.ckpt_keep)

    except KeyboardInterrupt:
        print("Interrupted! Saving emergency checkpoint...")
        save_checkpoint(train_cfg.save_dir / "sft_emergency.pkl", last_step + 1, model, optimizer)

    save_checkpoint(train_cfg.save_dir / "sft_last.pkl", last_step + 1, model, optimizer)
    logger.close()
    print("SFT training completed!")


if __name__ == "__main__":
    main()
