#!/usr/bin/env python
"""
Download and materialize a small text corpus for tokenizer training or small-scale pretraining.
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="allenai/c4", help="HF dataset name.")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--text-field", type=str, default="text")
    p.add_argument("--samples", type=int, default=50000)
    p.add_argument("--output", type=Path, default=Path("data/corpus.txt"))
    p.add_argument("--cache-dir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    ds = load_dataset(args.dataset, split=args.split, streaming=True, cache_dir=args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(tqdm(ds, total=args.samples)):
            if idx >= args.samples:
                break
            text = row.get(args.text_field) or row.get("content") or row.get("text") or ""
            if not text:
                continue
            f.write(text.replace("\n", " ").strip() + "\n")
    print(f"Wrote {args.samples} lines to {args.output}")


if __name__ == "__main__":
    main()
