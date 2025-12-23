#!/usr/bin/env python
"""
Prepare pre-tokenized SFT shards (npy pairs) directly from a HF dataset.

The script streams chat-style data, formats messages, tokenizes with SentencePiece,
and emits shard_*_x.npy / shard_*_y.npy files ready for MLX/MPS training.

Defaults target UltraChat-style data with minimal disk usage (no raw text saved).
"""
import argparse
from pathlib import Path
from typing import List

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from anarchobot.tokenizer import SentencePieceTokenizer


def format_messages(messages: List[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = str(msg.get("content", ""))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Create pretokenized SFT shards for MLX/MPS.")
    p.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    # UltraChat uses splits: train_sft/test_sft/train_gen/test_gen
    p.add_argument("--split", default="train_sft")
    p.add_argument("--text-field", default="messages")
    p.add_argument("--tokenizer", type=Path, required=True, help="SentencePiece tokenizer.model")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to write shard_*_x.npy/_y.npy")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--tokens-per-shard", type=int, default=1_000_000, help="Approx tokens per shard")
    p.add_argument("--max-shards", type=int, default=20, help="Cap number of shards to limit size")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--auth-token", default=None)
    args = p.parse_args()

    tokenizer = SentencePieceTokenizer(args.tokenizer)
    ds = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,
        cache_dir=args.cache_dir,
        token=args.auth_token,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    buffer: List[int] = []
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    tokens_in_shard = 0

    pbar = tqdm(desc="tokens", unit="tok")
    for sample in ds:
        if args.text_field not in sample:
            continue
        messages = sample[args.text_field]
        if not isinstance(messages, list):
            continue
        text = format_messages(messages)
        ids = tokenizer.encode(text)
        if not ids:
            continue
        buffer.extend(ids + [tokenizer.eos_id])
        tokens_in_shard += len(ids) + 1
        pbar.update(len(ids) + 1)

        while len(buffer) >= args.seq_len + 1:
            chunk = buffer[: args.seq_len + 1]
            buffer = buffer[args.seq_len :]
            xs.append(np.array(chunk[:-1], dtype=np.uint16))
            ys.append(np.array(chunk[1:], dtype=np.uint16))

        if tokens_in_shard >= args.tokens_per_shard:
            if xs:
                np.save(args.output_dir / f"shard_{shard_idx:05d}_x.npy", np.stack(xs))
                np.save(args.output_dir / f"shard_{shard_idx:05d}_y.npy", np.stack(ys))
            shard_idx += 1
            xs.clear()
            ys.clear()
            tokens_in_shard = 0
            if shard_idx >= args.max_shards:
                break

    # Flush tail
    if xs and shard_idx < args.max_shards:
        np.save(args.output_dir / f"shard_{shard_idx:05d}_x.npy", np.stack(xs))
        np.save(args.output_dir / f"shard_{shard_idx:05d}_y.npy", np.stack(ys))

    pbar.close()
    print(f"✅ Wrote {min(shard_idx + 1, args.max_shards)} shards to {args.output_dir}")


if __name__ == "__main__":
    main()
