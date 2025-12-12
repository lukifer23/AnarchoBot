#!/usr/bin/env python
"""
Stream a HuggingFace dataset split to a local text file with filtering and progress feedback.
Examples:
  python scripts/stream_text_dataset.py --dataset EliMC/Ultra-FineWeb --split en --text-field content \
    --samples 200000 --score-field score --score-min 0.8 --output data/ultrafineweb_en.txt
"""
import argparse
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="HF dataset name, e.g. EliMC/Ultra-FineWeb")
    p.add_argument("--config", default=None, help="Dataset config name (if any)")
    p.add_argument("--split", default="train", help="Dataset split to stream")
    p.add_argument("--text-field", required=True, help="Field containing text")
    p.add_argument("--samples", type=int, default=100000, help="Number of rows to write")
    p.add_argument("--score-field", default=None, help="Optional numeric score field for filtering")
    p.add_argument("--score-min", type=float, default=None, help="Keep rows with score >= score-min")
    p.add_argument("--max-chars", type=int, default=12000, help="Truncate overly long documents")
    p.add_argument("--output", type=Path, required=True, help="Output text file path")
    p.add_argument("--cache-dir", type=Path, default=None, help="Optional HF cache dir")
    p.add_argument("--auth-token", default=None, help="HF token (or set HF_TOKEN env var)")
    return p.parse_args()


def main():
    args = parse_args()
    auth = args.auth_token or os.environ.get("HF_TOKEN")
    ds = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
        cache_dir=args.cache_dir,
        token=auth,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as f:
        pbar = tqdm(total=args.samples, desc="streaming", unit="rows")
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
            f.write(text + "\n")
            written += 1
            pbar.update(1)
            if written >= args.samples:
                break
        pbar.close()
    print(f"Wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
