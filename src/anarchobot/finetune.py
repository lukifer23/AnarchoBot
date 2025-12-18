import argparse
import math
import signal
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_yaml_config
from .data import TokenChunkDataset, collate_batch
from .model import TransformerLM
from .optim import build_muon_adam_optimizer
from .tokenizer import SentencePieceTokenizer
from .utils import append_log, cosine_lr, get_device, load_checkpoint, rotate_checkpoints, save_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Supervised fine-tuning for AnarchoBot.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True, help="Pretrained checkpoint path.")
    return p.parse_args()


def main():
    args = parse_args()
    model_cfg, data_cfg, train_cfg = load_yaml_config(args.config)
    set_seed(123)

    tokenizer = SentencePieceTokenizer(train_cfg.tokenizer_path)
    device = get_device()
    if device.type != "mps":
        raise RuntimeError(f"Expected MPS device, got {device}")
    model = TransformerLM(model_cfg).to(device)
    model.enable_gradient_checkpointing(train_cfg.gradient_checkpointing)
    if train_cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    if train_cfg.optimizer == "muon_adam":
        optimizer = build_muon_adam_optimizer(
            model=model,
            lr_muon=train_cfg.lr,
            lr_adam=train_cfg.lr * train_cfg.adam_lr_multiplier,
            weight_decay=train_cfg.weight_decay,
            momentum=0.95,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    start_step = load_checkpoint(model, optimizer, args.base_checkpoint)
    print(f"Loaded base checkpoint {args.base_checkpoint} at step {start_step}")

    dataset = TokenChunkDataset(
        dataset=data_cfg.dataset,
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
    if train_cfg.warmup_steps < 1:
        raise ValueError("warmup_steps must be >= 1")
    model.train()
    pbar = tqdm(range(train_cfg.total_steps), total=train_cfg.total_steps)
    accum = train_cfg.grad_accum_steps
    log_path = train_cfg.log_path or (train_cfg.save_dir / "sft_log.jsonl")
    stop_requested = {"flag": False}

    def _handle_signal(signum, frame):
        stop_requested["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    tokens_processed = 0
    wall_start = time.perf_counter()

    for step in pbar:
        if stop_requested["flag"]:
            emergency = train_cfg.save_dir / "sft_emergency.pt"
            save_checkpoint(model, optimizer, step, emergency)
            rotate_checkpoints(train_cfg.save_dir, "sft_step_", train_cfg.ckpt_keep)
            break

        optimizer.zero_grad()
        total_loss = 0.0
        step_tokens = 0
        for _ in range(accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)
            x = x.to(device)
            y = y.to(device)
            step_tokens += x.numel()
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                _, loss = model(x, y)
                loss = loss / accum
            if device.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.item()

        if device.type == "cuda":
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)

        lr = cosine_lr(step + 1, train_cfg.warmup_steps, train_cfg.total_steps, train_cfg.lr, train_cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr if pg.get("use_muon", True) else lr * 1.5

        if device.type == "cuda":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        tokens_processed += step_tokens
        elapsed = time.perf_counter() - wall_start
        tokens_per_sec = tokens_processed / max(elapsed, 1e-6)
        ppl = math.exp(total_loss) if total_loss < 20 else float("inf")
        pbar.set_description(f"sft loss {total_loss:.4f} ppl {ppl:.2f} lr {lr:.6f} tok/s {tokens_per_sec:,.0f}")

        if step % train_cfg.log_interval == 0:
            append_log(
                log_path,
                {
                    "step": step,
                    "loss": total_loss,
                    "ppl": ppl,
                    "lr": lr,
                    "tokens_processed": tokens_processed,
                    "tokens_per_sec": tokens_per_sec,
                    "elapsed_sec": elapsed,
                },
            )

        if step > 0 and step % train_cfg.ckpt_interval == 0:
            ckpt = train_cfg.save_dir / f"sft_step_{step}.pt"
            save_checkpoint(model, optimizer, step, ckpt)
            rotate_checkpoints(train_cfg.save_dir, "sft_step_", train_cfg.ckpt_keep)

    if not stop_requested["flag"]:
        save_checkpoint(model, optimizer, train_cfg.total_steps, train_cfg.save_dir / "sft_last.pt")
        rotate_checkpoints(train_cfg.save_dir, "sft_step_", train_cfg.ckpt_keep)


if __name__ == "__main__":
    main()
