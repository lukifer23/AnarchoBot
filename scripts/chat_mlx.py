#!/usr/bin/env python
import argparse
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_map

from anarchobot.config import load_yaml_config
from anarchobot.tokenizer import SentencePieceTokenizer
from anarchobot_mlx.checkpointing import load_checkpoint
from anarchobot_mlx.model_mlx import TransformerLM


def parse_args():
    p = argparse.ArgumentParser(description="Chat with an MLX checkpoint.")
    p.add_argument("--config", type=Path, required=True, help="Model/training config YAML.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint .pkl from MLX training.")
    p.add_argument("--system-prompt", default="You are AnarchoBot.", help="System prompt to seed the chat.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    return p.parse_args()


def _cast_model_precision(model, precision: str):
    prec = precision.lower()
    target = None
    if prec in ("float16", "fp16", "half"):
        target = mx.float16
    elif prec in ("bfloat16", "bf16"):
        target = mx.bfloat16
    if target is None:
        return

    def cast_fn(x):
        if isinstance(x, mx.array) and mx.issubdtype(x.dtype, mx.floating):
            return x.astype(target)
        return x

    params = model.parameters()
    model.update(tree_map(cast_fn, params))


def generate_reply(model: TransformerLM, tokenizer: SentencePieceTokenizer, prompt: str, max_new_tokens: int, temperature: float, top_k: int):
    tokens = mx.array([tokenizer.encode(prompt)], dtype=mx.int32)
    out = model.generate(tokens, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


def main():
    args = parse_args()
    model_cfg, _, train_cfg = load_yaml_config(args.config)

    mx.set_default_device(mx.gpu)
    print(f"Using MLX device: {mx.default_device()}")

    model = TransformerLM(
        vocab_size=model_cfg.vocab_size,
        n_layers=model_cfg.n_layers,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        mlp_multiple=model_cfg.mlp_multiple,
        dropout=model_cfg.dropout,
        max_seq_len=model_cfg.max_seq_len,
        rope_theta=model_cfg.rope_theta,
        ffn_activation=model_cfg.ffn_activation,
        norm_eps=model_cfg.norm_eps,
        tie_embeddings=model_cfg.tie_embeddings,
    )
    _cast_model_precision(model, getattr(train_cfg, "precision", "float16"))

    tokenizer = SentencePieceTokenizer(train_cfg.tokenizer_path)
    load_checkpoint(args.checkpoint, model, optimizer=None, load_optimizer=False)

    history = [args.system_prompt]
    print("Entering MLX chat. Ctrl+C to exit.")
    while True:
        try:
            user_in = input("user> ")
        except KeyboardInterrupt:
            print("\nbye")
            break

        history.append(f"USER: {user_in}")
        prompt = "\n".join(history) + "\nASSISTANT:"
        reply_full = generate_reply(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        assistant_reply = reply_full[len(prompt) :].strip()
        history.append(f"ASSISTANT: {assistant_reply}")
        print(f"bot> {assistant_reply}")


if __name__ == "__main__":
    main()
