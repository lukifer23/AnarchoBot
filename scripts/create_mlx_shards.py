#!/usr/bin/env python
"""
Create pre-tokenized shards for MLX training (optimal performance).
Converts raw text shards to compressed integer token files.
"""
import argparse
import pickle
from pathlib import Path
from typing import List

import numpy as np
import mlx.core as mx
from tqdm import tqdm

from anarchobot.tokenizer import SentencePieceTokenizer


def create_mlx_shard(
    input_shard: Path,
    output_shard: Path,
    tokenizer: SentencePieceTokenizer,
    seq_len: int,
    compression: str = "npz",
):
    """
    Convert a raw text shard to pre-tokenized format.
    - compression=pickle: appends pickled {"x","y"} records (slow, compatible)
    - compression=npz: writes one compressed npz with stacked x/y arrays (fast)
    """
    buffer: List[int] = []
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    print(f"Processing {input_shard} -> {output_shard}")

    with input_shard.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            text = line.strip()
            if not text:
                continue

            ids = tokenizer.encode(text)
            buffer.extend(ids + [tokenizer.eos_id])

            while len(buffer) >= seq_len + 1:
                chunk = buffer[: seq_len + 1]
                buffer = buffer[seq_len:]
                xs.append(np.array(chunk[:-1], dtype=np.int32))
                ys.append(np.array(chunk[1:], dtype=np.int32))

    # tail
    if buffer and len(buffer) >= seq_len + 1:
        chunk = buffer[: seq_len + 1]
        xs.append(np.array(chunk[:-1], dtype=np.int32))
        ys.append(np.array(chunk[1:], dtype=np.int32))

    if not xs:
        print(f"Skipping empty shard {input_shard}")
        return

    output_shard.parent.mkdir(parents=True, exist_ok=True)
    if compression == "pickle":
        # Use write mode (not append) to create new file per shard
        with output_shard.open("wb") as f_out:
            for x, y in zip(xs, ys):
                pickle.dump({"x": mx.array(x), "y": mx.array(y)}, f_out)
    elif compression == "npz":
        x_arr = np.stack(xs)
        y_arr = np.stack(ys)
        np.savez_compressed(str(output_shard), x=x_arr, y=y_arr)
    else:
        raise ValueError(f"Unknown compression: {compression}")

    print(f"Completed {input_shard}: sequences {len(xs)}")


def main():
    parser = argparse.ArgumentParser(description="Create pre-tokenized MLX shards")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with raw text shards")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for MLX shards")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Tokenizer model file")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--compression", choices=["pickle", "npz"], default="npz", help="Compression format")
    parser.add_argument("--max-shards", type=int, help="Limit number of shards to process")
    parser.add_argument("--shard-offset", type=int, default=0, help="Start processing from this shard index")
    args = parser.parse_args()

    tokenizer = SentencePieceTokenizer(args.tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all input shards
    input_shards = sorted(args.input_dir.glob("shard_*.txt"))
    # Apply offset
    input_shards = input_shards[args.shard_offset:]
    if args.max_shards:
        input_shards = input_shards[:args.max_shards]

    print(f"Found {len(input_shards)} input shards")
    print(f"Output directory: {args.output_dir}")
    print(f"Using tokenizer: {args.tokenizer}")

    for i, input_shard in enumerate(tqdm(input_shards, desc="Processing shards")):
        suffix = "npz" if args.compression == "npz" else "mlx"
        output_shard = args.output_dir / f"shard_{i + args.shard_offset:05d}.{suffix}"
        create_mlx_shard(
            input_shard=input_shard,
            output_shard=output_shard,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            compression=args.compression
        )

    print(f"\n✅ Created {len(input_shards)} MLX shards in {args.output_dir}")


if __name__ == "__main__":
    main()
