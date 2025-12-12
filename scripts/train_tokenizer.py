#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anarchobot.tokenizer import SentencePieceTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/corpus.txt"), help="Text corpus file.")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--model-prefix", type=Path, default=Path("data/tokenizer"))
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = SentencePieceTokenizer.train(
        input_paths=[args.input],
        vocab_size=args.vocab_size,
        model_prefix=args.model_prefix,
    )
    print(f"Trained tokenizer at {tokenizer.model_file} with vocab {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
