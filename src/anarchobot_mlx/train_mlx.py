import argparse
import math
import time
from pathlib import Path
import pickle

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten
import numpy as np

from anarchobot.config import load_yaml_config
from anarchobot.utils import cosine_lr, rotate_checkpoints
from anarchobot.tokenizer import SentencePieceTokenizer
from anarchobot_mlx.data_mlx import StreamingShardLoader, token_chunk_iterator
from anarchobot_mlx.model_mlx import TransformerLM
from anarchobot.training_logger import TrainingLogger, TrainingMetrics


def parse_args():
    p = argparse.ArgumentParser(description="Pre-train AnarchoBot with MLX backend.")
    p.add_argument("--config", type=Path, required=True, help="YAML config file.")
    p.add_argument("--shard-dir", type=Path, required=True, help="Directory of shards.")
    p.add_argument("--format", choices=["mlx", "npz", "npy", "txt"], default="mlx",
                   help="Shard format: pretokenized mlx/npz/npy or raw txt.")
    p.add_argument("--profile-steps", type=int, default=0, help="If >0, capture per-step timings for first N steps.")
    p.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from.")
    return p.parse_args()


def safe_mx_convert(x):
    """Convert numpy arrays back to MLX arrays, handling dtype issues"""
    if isinstance(x, np.ndarray):
        if x.dtype == np.float32:
            return mx.array(x, dtype=mx.float32)
        elif x.dtype == np.float16:
            return mx.array(x, dtype=mx.bfloat16) # MLX prefers bf16/f32
        else:
            return mx.array(x)
    return x

def load_checkpoint(path: Path, model: nn.Module, optimizer: optim.Optimizer):
    print(f"Loading checkpoint from {path}...")
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            payload = pickle.load(f)
        
        # Load model params
        params = tree_map(safe_mx_convert, payload["params"])
        model.update(params)
        
        # Load optimizer state if available
        # Note: MLX optimizer state structure might differ from custom implementation
        # We try to load it if it matches, otherwise we reset
        if "opt_state" in payload:
            try:
                opt_state = tree_map(safe_mx_convert, payload["opt_state"])
                optimizer.state = opt_state
                print("Optimizer state loaded.")
            except Exception as e:
                print(f"⚠️  Could not load optimizer state: {e}. Starting optimizer from scratch.")

        return int(payload.get("step", 0))
        
    print(f"⚠️  Skipping legacy/unsupported checkpoint {path}.")
    return 0


def save_checkpoint(path: Path, step: int, model: nn.Module, optimizer: optim.Optimizer):
    def safe_numpy_convert(x):
        if isinstance(x, mx.array):
            if x.dtype == mx.bfloat16:
                return np.array(x.astype(mx.float32))
            return np.array(x)
        return x

    payload = {
        "step": step,
        "params": tree_map(safe_numpy_convert, model.parameters()),
        "opt_state": tree_map(safe_numpy_convert, optimizer.state)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Saved checkpoint to {path}")


def main():
    # Clear Python cache
    import subprocess, sys
    try: subprocess.run([sys.executable, "-m", "py_compile", __file__], check=True, capture_output=True)
    except: pass

    args = parse_args()
    model_cfg, data_cfg, train_cfg = load_yaml_config(args.config)

    # Device setup
    mx.set_default_device(mx.gpu)
    print(f"Using MLX device: {mx.default_device()}")

    # Model
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
    
    # Precision
    prec = getattr(train_cfg, "precision", "float16")
    target_dtype = None
    if prec in ["float16", "fp16", "half"]:
        target_dtype = mx.float16
    elif prec in ["bfloat16", "bf16"]:
        target_dtype = mx.bfloat16
        
    if target_dtype is not None:
        def cast_fn(x):
            if isinstance(x, mx.array) and mx.issubdtype(x.dtype, mx.floating):
                return x.astype(target_dtype)
            return x
        # Cast all parameters
        params = tree_map(cast_fn, model.parameters())
        model.update(params)

    # Optimizer (Muon not available in MLX; use AdamW with config-aware LR)
    if getattr(train_cfg, "optimizer", "adamw") != "adamw":
        print("Muon is not available on MLX; falling back to AdamW for MLX backend.")
    optimizer = optim.AdamW(
        learning_rate=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    
    # Resume
    start_step = 0
    if args.resume and args.resume.exists():
        start_step = load_checkpoint(args.resume, model, optimizer)
    elif train_cfg.checkpoint_path and train_cfg.checkpoint_path.exists():
         start_step = load_checkpoint(train_cfg.checkpoint_path, model, optimizer)
    else:
        last_ckpt = train_cfg.save_dir / "mlx_last.pkl"
        if last_ckpt.exists():
            start_step = load_checkpoint(last_ckpt, model, optimizer)

    # Data
    print(f"Loading data from {args.shard_dir} ({args.format})...")
    try:
        tokenizer = SentencePieceTokenizer(train_cfg.tokenizer_path)
    except Exception as e:
        print(f"⚠️  Could not load tokenizer from {train_cfg.tokenizer_path}: {e}")
        print("    Continuing without tokenizer (text logging will be disabled).")
        tokenizer = None

    seq_len = data_cfg.seq_len
    batch_size = train_cfg.micro_batch_size
    shard_format = args.format

    if shard_format == "txt":
        if tokenizer is None:
            raise ValueError("Tokenizer required for txt format.")
        data_iter = iter(token_chunk_iterator(args.shard_dir, tokenizer, seq_len, format="txt"))
    else:
        data_loader = StreamingShardLoader(args.shard_dir, batch_size, seq_len, shard_format)
        data_iter = iter(data_loader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            print("Restarting data iterator...")
            if shard_format == "txt":
                if tokenizer is None:
                    raise ValueError("Tokenizer required for txt format.")
                data_iter = iter(token_chunk_iterator(args.shard_dir, tokenizer, seq_len, "txt"))
            else:
                data_loader = StreamingShardLoader(args.shard_dir, batch_size, seq_len, shard_format)
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
            if grad_accum is None:
                grad_accum = grads
            else:
                grad_accum = tree_map(lambda a, b: a + b, grad_accum, grads)
        grad_accum = tree_map(lambda g: g / accum, grad_accum)
        optimizer.update(model, grad_accum)
        mx.eval(model.parameters(), optimizer.state)
        return total_loss / accum, tokens

    if getattr(train_cfg, "compile", False):
        accumulate_step = mx.compile(accumulate_step)

    logger = TrainingLogger(
        log_dir=train_cfg.save_dir / "logs",
        experiment_name=f"mlx_pretrain_{model_cfg.d_model}d_{model_cfg.n_layers}l",
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

    print("Starting training loop...")
    accum = max(1, train_cfg.grad_accum_steps)

    xb, yb = next_batch()
    warm_loss, _ = loss_and_grad_fn(model, xb, yb)
    mx.eval(warm_loss)
    print(f"Warmup forward pass loss: {warm_loss.item():.4f}")

    wall_start = time.perf_counter()
    tokens_processed = 0

    try:
        for step in range(start_step, train_cfg.total_steps):
            step_start = time.perf_counter()

            lr = cosine_lr(step + 1, train_cfg.warmup_steps, train_cfg.total_steps, train_cfg.lr, train_cfg.min_lr)
            optimizer.learning_rate = lr

            loss_val, step_tokens = accumulate_step(accum)

            step_time = time.perf_counter() - step_start
            tokens_processed += step_tokens
            total_elapsed = time.perf_counter() - wall_start

            tps = step_tokens / max(step_time, 1e-6)
            tps_avg = tokens_processed / max(total_elapsed, 1e-6)

            if step % train_cfg.log_interval == 0:
                print(f"Step {step}: loss {loss_val:.4f}, lr {lr:.6f}, tok/s {tps:.0f} (avg {tps_avg:.0f})")

            logger.log_step(
                TrainingMetrics(
                    step=step,
                    loss=loss_val,
                    learning_rate=lr,
                    grad_norm=0.0,
                    tokens_processed=step_tokens,
                    throughput_tokens_per_sec=tps,
                    step_time_sec=step_time,
                    timestamp=time.time(),
                    perplexity=math.exp(loss_val) if loss_val < 20 else float("inf"),
                )
            )

            if step > 0 and step % train_cfg.ckpt_interval == 0:
                save_checkpoint(train_cfg.save_dir / f"mlx_step_{step}.pkl", step, model, optimizer)
                rotate_checkpoints(train_cfg.save_dir, "mlx_step_", train_cfg.ckpt_keep)

    except KeyboardInterrupt:
        print("Interrupted! Saving emergency checkpoint...")
        save_checkpoint(train_cfg.save_dir / "mlx_emergency.pkl", step, model, optimizer)

    save_checkpoint(train_cfg.save_dir / "mlx_last.pkl", train_cfg.total_steps, model, optimizer)
    logger.close()

if __name__ == "__main__":
    main()
