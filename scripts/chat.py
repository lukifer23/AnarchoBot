#!/usr/bin/env python
import argparse
from pathlib import Path

import torch

from anarchobot.config import ModelConfig
from anarchobot.generation import chat_loop
from anarchobot.model import TransformerLM
from anarchobot.tokenizer import SentencePieceTokenizer
from anarchobot.utils import get_device, load_checkpoint
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Chat with a trained AnarchoBot checkpoint.")
    p.add_argument("--config", type=Path, required=True, help="Model config YAML.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt file.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    model_cfg = ModelConfig(vocab_size=cfg["model"]["vocab_size"], **{k: v for k, v in cfg["model"].items() if k != "vocab_size"})
    tokenizer = SentencePieceTokenizer(Path(cfg["train"]["tokenizer_path"]))
    device = get_device()
    if device.type != "mps":
        raise RuntimeError(f"Expected MPS device, got {device}")
    model = TransformerLM(model_cfg).to(device)
    load_checkpoint(model, None, args.checkpoint)
    chat_loop(model, tokenizer)


if __name__ == "__main__":
    main()
