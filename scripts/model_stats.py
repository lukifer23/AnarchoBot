#!/usr/bin/env python
import argparse
from pathlib import Path

import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anarchobot.config import ModelConfig
from anarchobot.model import TransformerLM


def parse_args():
    p = argparse.ArgumentParser(description="Report parameter count and token budget.")
    p.add_argument("--config", type=Path, required=True, help="Model config YAML.")
    p.add_argument("--token-ratio", type=float, default=20.0, help="Tokens-per-parameter ratio (Chinchilla-style).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    model_cfg = ModelConfig(vocab_size=cfg["model"]["vocab_size"], **{k: v for k, v in cfg["model"].items() if k != "vocab_size"})
    model = TransformerLM(model_cfg)
    params = sum(p.numel() for p in model.parameters())
    billion_params = params / 1e9
    target_tokens = params * args.token_ratio
    print(f"Model parameters: {params:,} ({billion_params:.3f}B)")
    print(f"Recommended tokens (ratio {args.token_ratio}): {int(target_tokens):,} ({target_tokens/1e9:.3f}B tokens)")


if __name__ == "__main__":
    main()
