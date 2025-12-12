import argparse
import math
import signal
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from .config import DataConfig, ModelConfig, TrainingConfig
from .data import TokenChunkDataset, collate_batch
from .memory_monitor import MemoryMonitor, optimize_batch_size_for_memory
from .model import TransformerLM
from .optim import build_muon_adam_optimizer
from .tokenizer import SentencePieceTokenizer
from .training_logger import TrainingLogger, TrainingMetrics, TrainingProgressTracker
from .utils import cosine_lr, get_device, load_checkpoint, rotate_checkpoints, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train AnarchoBot from scratch.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file.")
    return parser.parse_args()


def load_configs(path: Path):
    cfg = yaml.safe_load(path.read_text())

    # Load model config with proper type conversion
    model_dict = dict(cfg["model"])
    # Ensure numeric types are properly converted
    model_numeric_fields = ["mlp_multiple", "dropout", "rope_theta", "norm_eps"]
    for field in model_numeric_fields:
        if field in model_dict and isinstance(model_dict[field], str):
            model_dict[field] = float(model_dict[field])
    model_cfg = ModelConfig(**model_dict)

    # Load data config
    data_dict = dict(cfg["data"])
    if data_dict.get("cache_dir"):
        data_dict["cache_dir"] = Path(data_dict["cache_dir"])
    data_cfg = DataConfig(**data_dict)

    # Load training config
    train_dict = dict(cfg["train"])
    for key in ["save_dir", "tokenizer_path", "checkpoint_path", "log_path"]:
        if train_dict.get(key):
            train_dict[key] = Path(train_dict[key])

    # Ensure numeric types are properly converted
    train_numeric_fields = ["lr", "min_lr", "weight_decay", "max_grad_norm"]
    for field in train_numeric_fields:
        if field in train_dict and isinstance(train_dict[field], str):
            train_dict[field] = float(train_dict[field])
    if "ckpt_keep" in train_dict and isinstance(train_dict["ckpt_keep"], str):
        train_dict["ckpt_keep"] = int(train_dict["ckpt_keep"])

    train_cfg = TrainingConfig(**train_dict)
    return model_cfg, data_cfg, train_cfg


def main():
    args = parse_args()
    model_cfg, data_cfg, train_cfg = load_configs(args.config)
    set_seed(42)

    tokenizer = SentencePieceTokenizer(train_cfg.tokenizer_path)
    device = get_device()
    if device.type != "mps":
        raise RuntimeError(f"Expected MPS device, got {device}")
    print(f"Using device: {device}")

    # Initialize memory monitor and check batch size
    memory_monitor = MemoryMonitor(device)
    print("🔍 Checking memory requirements...")
    memory_estimate = memory_monitor.estimate_training_memory(
        TransformerLM(model_cfg), train_cfg.micro_batch_size, data_cfg.seq_len
    )
    if memory_estimate["usage_percent"] > 80:
        print("⚠️  High memory usage predicted. Consider reducing batch size or enabling gradient checkpointing.")
        suggested_batch = optimize_batch_size_for_memory(
            TransformerLM(model_cfg), data_cfg.seq_len, target_memory_percent=0.7
        )
        if suggested_batch < train_cfg.micro_batch_size:
            print(f"💡 Suggested batch size: {suggested_batch} (current: {train_cfg.micro_batch_size})")

    model = TransformerLM(model_cfg).to(device)
    model.enable_gradient_checkpointing(train_cfg.gradient_checkpointing)
    if train_cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = build_muon_adam_optimizer(
        model=model,
        lr_muon=train_cfg.lr,
        lr_adam=train_cfg.lr * 1.5,
        weight_decay=train_cfg.weight_decay,
        momentum=0.95,
    )

    start_step = 0
    if train_cfg.checkpoint_path and train_cfg.checkpoint_path.exists():
        start_step = load_checkpoint(model, optimizer, train_cfg.checkpoint_path)
        print(f"Resumed from step {start_step}")

    dataset = TokenChunkDataset(
        dataset=data_cfg.dataset,
        config=data_cfg.config,
        split=data_cfg.split,
        text_field=data_cfg.text_field,
        tokenizer=tokenizer,
        seq_len=data_cfg.seq_len,
        shuffle_buffer=data_cfg.shuffle_buffer,
        streaming=data_cfg.streaming,
        cache_dir=data_cfg.cache_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.micro_batch_size,
        collate_fn=collate_batch,
        num_workers=data_cfg.num_workers,
        pin_memory=False,
    )
    data_iter = iter(dataloader)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    amp_dtype = torch.bfloat16 if train_cfg.precision == "bfloat16" and device.type != "mps" else torch.float16
    total_steps = train_cfg.total_steps
    if train_cfg.warmup_steps < 1:
        raise ValueError("warmup_steps must be >= 1")
    model.train()

    # Initialize training logger and progress tracker
    logger = TrainingLogger(
        log_dir=train_cfg.save_dir / "logs",
        experiment_name=f"pretrain_{model_cfg.d_model}d_{model_cfg.n_layers}l",
        console_log_interval=train_cfg.log_interval
    )

    # Log hyperparameters (only simple types for TensorBoard compatibility)
    hyperparams = {
        "model_vocab_size": model_cfg.vocab_size,
        "model_n_layers": model_cfg.n_layers,
        "model_d_model": model_cfg.d_model,
        "model_n_heads": model_cfg.n_heads,
        "data_dataset": data_cfg.dataset,
        "data_seq_len": data_cfg.seq_len,
        "train_total_steps": train_cfg.total_steps,
        "train_batch_size": train_cfg.micro_batch_size,
        "train_lr": train_cfg.lr,
        "train_weight_decay": train_cfg.weight_decay,
        "device": str(device),
        "torch_version": torch.__version__
    }
    logger.log_hyperparameters(hyperparams)

    progress_tracker = TrainingProgressTracker(total_steps)

    pbar = tqdm(range(start_step, total_steps), initial=start_step, total=total_steps)
    accum = train_cfg.grad_accum_steps

    # Training loop variables
    tokens_processed_total = 0
    step_start_time = time.time()
    stop_requested = {"flag": False}

    def _handle_signal(signum, frame):
        stop_requested["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    for step in pbar:
        if stop_requested["flag"]:
            emergency = train_cfg.save_dir / "emergency.pt"
            save_checkpoint(model, optimizer, step, emergency)
            rotate_checkpoints(train_cfg.save_dir, "step_", train_cfg.ckpt_keep)
            break
        step_start_time = time.time()

        with memory_monitor.monitor_step(step):
            optimizer.zero_grad()
            total_loss = 0.0
            tokens_in_step = 0

            for micro in range(accum):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    x, y = next(data_iter)
                x = x.to(device)
                y = y.to(device)
                tokens_in_step += x.numel()

                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits, loss = model(x, y)
                    loss = loss / accum
                if device.type == "cuda":
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                total_loss += loss.item()

            if device.type == "cuda":
                scaler.unscale_(optimizer)

            # Calculate gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)

            lr = cosine_lr(step + 1, train_cfg.warmup_steps, total_steps, train_cfg.lr, train_cfg.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr if pg.get("use_muon", True) else lr * 1.5

            if device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        # Calculate metrics
        step_time = time.time() - step_start_time
        tokens_processed_total += tokens_in_step
        throughput = tokens_in_step / step_time if step_time > 0 else 0

        # Get memory stats
        memory_stats = memory_monitor.get_memory_stats(step)

        # Calculate perplexity
        perplexity = math.exp(total_loss) if total_loss < 20 else float("inf")

        # Calculate parameter norm (expensive, only occasionally)
        param_norm = None
        if step % (train_cfg.log_interval * 10) == 0:
            param_norm = torch.norm(torch.stack([p.norm() for p in model.parameters()])).item()

        # Create metrics object
        metrics = TrainingMetrics(
            step=step,
            loss=total_loss,
            learning_rate=lr,
            grad_norm=grad_norm.item(),
            tokens_processed=tokens_in_step,
            throughput_tokens_per_sec=throughput,
            step_time_sec=step_time,
            timestamp=time.time(),
            gpu_memory_used_mb=memory_stats.gpu_memory_used,
            gpu_memory_peak_mb=memory_stats.gpu_memory_peak,
            system_memory_percent=memory_stats.system_memory_percent,
            perplexity=perplexity,
            param_norm=param_norm
        )

        # Log metrics
        logger.log_step(metrics)

        # Update progress bar
        progress = progress_tracker.get_progress_summary(step)
        pbar.set_description(
            f"loss {total_loss:.4f} | ppl {perplexity:.2f} | lr {lr:.6f} | tok/s {throughput:,.0f}"
        )

        # Check early stopping
        if progress_tracker.should_stop_early(total_loss, step):
            print(f"🛑 Early stopping triggered at step {step} (no improvement for {progress_tracker.early_stopping_patience} steps)")
            break

        if step > 0 and step % train_cfg.ckpt_interval == 0:
            ckpt_path = train_cfg.save_dir / f"step_{step}.pt"
            save_checkpoint(model, optimizer, step, ckpt_path)
            rotate_checkpoints(train_cfg.save_dir, "step_", train_cfg.ckpt_keep)

    if not stop_requested["flag"]:
        save_checkpoint(model, optimizer, total_steps, train_cfg.save_dir / "last.pt")
        rotate_checkpoints(train_cfg.save_dir, "step_", train_cfg.ckpt_keep)

    # Final logging and cleanup
    logger.save_final_summary()
    memory_summary = memory_monitor.get_memory_summary()
    print("📊 Training completed!")
    print(f"   Final memory usage: {memory_summary}")

    # Close loggers
    logger.close()


if __name__ == "__main__":
    main()
