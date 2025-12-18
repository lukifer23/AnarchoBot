#!/usr/bin/env python
import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anarchobot.config import load_yaml_config
from anarchobot.model import TransformerLM


def parse_args():
    p = argparse.ArgumentParser(description="Report parameter count and token budget.")
    p.add_argument("--config", type=Path, required=True, help="Model config YAML.")
    p.add_argument("--token-ratio", type=float, default=20.0, help="Tokens-per-parameter ratio (Chinchilla-style).")
    return p.parse_args()


def main():
    args = parse_args()
    model_cfg, data_cfg, train_cfg = load_yaml_config(args.config)
    model = TransformerLM(model_cfg)
    params = sum(p.numel() for p in model.parameters())
    billion_params = params / 1e9
    target_tokens = params * args.token_ratio
    print(f"Model parameters: {params:,} ({billion_params:.3f}B)")
    print(f"Recommended tokens (ratio {args.token_ratio}): {int(target_tokens):,} ({target_tokens/1e9:.3f}B tokens)")
    if data_cfg and train_cfg:
        tokens_per_step = train_cfg.tokens_per_step(data_cfg.seq_len)
        print(f"Config tokens/step: {tokens_per_step:,}")
        print(f"Planned total tokens: {train_cfg.total_tokens(data_cfg.seq_len):,}")


if __name__ == "__main__":
    main()
