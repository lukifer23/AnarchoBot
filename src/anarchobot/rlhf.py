import argparse
import math
import signal
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from .config import load_yaml_config
from .model import TransformerLM
from .optim import build_muon_adam_optimizer
from .tokenizer import SentencePieceTokenizer
from .utils import append_log, cosine_lr, get_device, load_checkpoint, rotate_checkpoints, save_checkpoint, set_seed


class PreferenceDataset(IterableDataset):
    """
    Streams a preference dataset with `prompt`, `chosen`, and `rejected` fields.
    """

    def __init__(self, dataset: str, split: str, tokenizer: SentencePieceTokenizer, seq_len: int, cache_dir=None):
        super().__init__()
        from datasets import load_dataset

        self.ds = load_dataset(dataset, split=split, streaming=True, cache_dir=cache_dir)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def _concat(self, prompt: str, response: str):
        text = f"{prompt}\n{response}"
        ids = self.tokenizer.encode(text)[: self.seq_len - 1]
        ids.append(self.tokenizer.eos_id)
        if len(ids) < self.seq_len:
            ids.extend([self.tokenizer.pad_id] * (self.seq_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def __iter__(self):
        for row in self.ds:
            prompt = row.get("prompt") or row.get("question") or ""
            chosen = row.get("chosen") or row.get("chosen_response") or row.get("response_chosen") or ""
            rejected = row.get("rejected") or row.get("rejected_response") or row.get("response_rejected") or ""
            if not prompt or not chosen or not rejected:
                continue
            yield self._concat(prompt, chosen), self._concat(prompt, rejected)


def sequence_logprob(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logprobs = torch.log_softmax(logits, dim=-1)
    target_logp = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    mask = (target != 0).float()
    return (target_logp * mask).sum(dim=-1)


def parse_args():
    p = argparse.ArgumentParser(description="DPO-style preference tuning for AnarchoBot.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--sft-checkpoint", type=Path, required=True, help="Path to SFT model checkpoint.")
    p.add_argument("--beta", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    beta = args.beta
    model_cfg, data_cfg, train_cfg = load_yaml_config(args.config)
    set_seed(7)
    device = get_device()
    if device.type != "mps":
        raise RuntimeError(f"Expected MPS device, got {device}")
    tokenizer = SentencePieceTokenizer(train_cfg.tokenizer_path)

    policy = TransformerLM(model_cfg).to(device)
    policy.enable_gradient_checkpointing(train_cfg.gradient_checkpointing)
    if train_cfg.compile and hasattr(torch, "compile"):
        policy = torch.compile(policy)
    ref_model = TransformerLM(model_cfg).to(device)
    if train_cfg.optimizer == "muon_adam":
        optimizer = build_muon_adam_optimizer(
            model=policy,
            lr_muon=train_cfg.lr,
            lr_adam=train_cfg.lr * train_cfg.adam_lr_multiplier,
            weight_decay=train_cfg.weight_decay,
            momentum=0.95,
        )
    else:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    load_checkpoint(policy, None, args.sft_checkpoint)
    ref_model.load_state_dict(policy.state_dict())
    for p in ref_model.parameters():
        p.requires_grad_(False)

    dataset = PreferenceDataset(dataset=data_cfg.dataset, split=data_cfg.split, tokenizer=tokenizer, seq_len=data_cfg.seq_len)
    dataloader = DataLoader(dataset, batch_size=train_cfg.micro_batch_size, num_workers=data_cfg.num_workers)
    data_iter = iter(dataloader)

    pbar = tqdm(range(train_cfg.total_steps), total=train_cfg.total_steps)
    accum = train_cfg.grad_accum_steps
    policy.train()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    amp_dtype = torch.bfloat16 if train_cfg.precision == "bfloat16" and device.type != "mps" else torch.float16
    if train_cfg.warmup_steps < 1:
        raise ValueError("warmup_steps must be >= 1")
    log_path = train_cfg.log_path or (train_cfg.save_dir / "dpo_log.jsonl")
    stop_requested = {"flag": False}

    def _handle_signal(signum, frame):
        stop_requested["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    tokens_processed = 0
    wall_start = time.perf_counter()

    for step in pbar:
        if stop_requested["flag"]:
            emergency = train_cfg.save_dir / "dpo_emergency.pt"
            save_checkpoint(policy, optimizer, step, emergency)
            rotate_checkpoints(train_cfg.save_dir, "dpo_step_", train_cfg.ckpt_keep)
            break

        optimizer.zero_grad()
        total_loss = 0.0
        step_tokens = 0
        for _ in range(accum):
            try:
                chosen, rejected = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                chosen, rejected = next(data_iter)
            chosen = chosen.to(device)
            rejected = rejected.to(device)
            step_tokens += chosen.numel() + rejected.numel()

            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                c_logits, _ = policy(chosen[:, :-1], chosen[:, 1:])
                r_logits, _ = policy(rejected[:, :-1], rejected[:, 1:])
                with torch.no_grad():
                    rc_logits, _ = ref_model(chosen[:, :-1], chosen[:, 1:])
                    rr_logits, _ = ref_model(rejected[:, :-1], rejected[:, 1:])
                logp_c = sequence_logprob(c_logits, chosen[:, 1:])
                logp_r = sequence_logprob(r_logits, rejected[:, 1:])
                logp_c_ref = sequence_logprob(rc_logits, chosen[:, 1:])
                logp_r_ref = sequence_logprob(rr_logits, rejected[:, 1:])
                advantages = beta * ((logp_c - logp_r) - (logp_c_ref - logp_r_ref))
                loss = -torch.mean(torch.log(torch.sigmoid(advantages)))
                loss = loss / accum

            if device.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.item()

        if device.type == "cuda":
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), train_cfg.max_grad_norm)

        lr = cosine_lr(step + 1, train_cfg.warmup_steps, train_cfg.total_steps, train_cfg.lr, train_cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if device.type == "cuda":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        tokens_processed += step_tokens
        elapsed = time.perf_counter() - wall_start
        tokens_per_sec = tokens_processed / max(elapsed, 1e-6)
        pbar.set_description(f"dpo loss {total_loss:.4f} lr {lr:.6f} tok/s {tokens_per_sec:,.0f}")

        if step % train_cfg.log_interval == 0:
            append_log(
                log_path,
                {
                    "step": step,
                    "loss": total_loss,
                    "lr": lr,
                    "tokens_processed": tokens_processed,
                    "tokens_per_sec": tokens_per_sec,
                    "elapsed_sec": elapsed,
                },
            )

        if step > 0 and step % train_cfg.ckpt_interval == 0:
            save_checkpoint(policy, optimizer, step, train_cfg.save_dir / f"dpo_step_{step}.pt")
            rotate_checkpoints(train_cfg.save_dir, "dpo_step_", train_cfg.ckpt_keep)

    if not stop_requested["flag"]:
        save_checkpoint(policy, optimizer, train_cfg.total_steps, train_cfg.save_dir / "dpo_last.pt")
        rotate_checkpoints(train_cfg.save_dir, "dpo_step_", train_cfg.ckpt_keep)


if __name__ == "__main__":
    main()
