#!/usr/bin/env python
"""
Stream a HF dataset until a target token budget is reached, writing text shards.
Requires an existing tokenizer model (SentencePiece).

Example (≈2.8B tokens target):
python scripts/stream_for_token_budget.py \
  --dataset EliMC/Ultra-FineWeb --split en --text-field content \
  --tokenizer data/tokenizer.model --target-tokens 2800000000 \
  --tokens-per-shard 50000000 --score-field score --score-min 0.8 \
  --output-dir data/ultrafineweb_full
"""
import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anarchobot.tokenizer import SentencePieceTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Stream dataset to reach a token budget.")
    p.add_argument("--dataset", required=True, help="HF dataset name, e.g. EliMC/Ultra-FineWeb")
    p.add_argument("--config", default=None, help="Dataset config (if any)")
    p.add_argument("--split", default="train", help="Dataset split")
    p.add_argument("--text-field", required=True, help="Field containing text")
    p.add_argument("--tokenizer", type=Path, required=True, help="SentencePiece tokenizer model path")
    p.add_argument("--target-tokens", type=int, required=True, help="Total tokens to collect (e.g., 2800000000)")
    p.add_argument("--tokens-per-shard", type=int, default=50_000_000, help="Token cap per output shard")
    p.add_argument("--score-field", default=None, help="Optional numeric score field")
    p.add_argument("--score-min", type=float, default=None, help="Keep rows with score >= score-min")
    p.add_argument("--max-chars", type=int, default=12000, help="Trim overly long documents")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory for shards")
    p.add_argument("--cache-dir", type=Path, default=None, help="HF cache dir")
    p.add_argument("--auth-token", default=None, help="HF token or set HF_TOKEN env var")
    return p.parse_args()


def open_shard(out_dir: Path, shard_idx: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"shard_{shard_idx:05d}.txt"
    return shard_path, shard_path.open("w", encoding="utf-8")


def main():
    args = parse_args()
    auth = args.auth_token or os.environ.get("HF_TOKEN")
    tokenizer = SentencePieceTokenizer(args.tokenizer)
    ds = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
        cache_dir=args.cache_dir,
        token=auth,
    )

    total_tokens = 0
    total_rows = 0
    shard_tokens = 0
    shard_idx = 0
    shard_path, shard_file = open_shard(args.output_dir, shard_idx)

    pbar = tqdm(total=args.target_tokens, desc="tokens", unit="tok")
    for row in ds:
        if args.text_field not in row:
            continue
        if args.score_field and args.score_field in row and args.score_min is not None:
            try:
                if float(row[args.score_field]) < args.score_min:
                    continue
            except Exception:
                continue

        text = row[args.text_field]
        if not isinstance(text, str):
            continue
        text = text.replace("\n", " ").strip()
        if not text:
            continue
        if args.max_chars and len(text) > args.max_chars:
            text = text[: args.max_chars]

        tok_ids = tokenizer.encode(text)
        tok_len = len(tok_ids)
        if tok_len == 0:
            continue

        shard_file.write(text + "\n")
        total_tokens += tok_len
        shard_tokens += tok_len
        total_rows += 1
        pbar.update(tok_len)

        if shard_tokens >= args.tokens_per_shard:
            shard_file.close()
            shard_idx += 1
            shard_tokens = 0
            shard_path, shard_file = open_shard(args.output_dir, shard_idx)

        if total_tokens >= args.target_tokens:
            break

    pbar.close()
    shard_file.close()
    print(f"Wrote {total_rows} rows across {shard_idx + 1} shards to {args.output_dir}")
    print(f"Total tokens (approx): {total_tokens:,}")


if __name__ == "__main__":
    main()
