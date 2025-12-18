#!/usr/bin/env python
"""
Unified pre-train data pipeline for both MPS (PyTorch) and MLX backends.

Steps:
1) plan      - report token budget + shard counts from a config
2) stream    - stream HF dataset to raw text shards until target tokens reached
3) tokenize  - convert text shards to pretokenized npy/npz shards (X/Y pairs)
4) cleanup   - remove intermediate shards to free disk

No downloads or training are performed automatically; invoke subcommands explicitly.
"""
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from datasets import load_dataset
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anarchobot.config import load_yaml_config
from anarchobot.tokenizer import SentencePieceTokenizer


def _compute_plan(config: Path, tokens_per_shard: int):
    model_cfg, data_cfg, train_cfg = load_yaml_config(config)
    tokens_per_step = train_cfg.tokens_per_step(data_cfg.seq_len)
    total_tokens = train_cfg.total_tokens(data_cfg.seq_len)
    shards_needed = math.ceil(total_tokens / tokens_per_shard)
    return {
        "tokens_per_step": tokens_per_step,
        "total_tokens": total_tokens,
        "tokens_per_shard": tokens_per_shard,
        "shards_needed": shards_needed,
        "seq_len": data_cfg.seq_len,
        "dataset": data_cfg.dataset,
        "text_field": data_cfg.text_field,
    }


def plan_cmd(args):
    summary = _compute_plan(args.config, args.tokens_per_shard)
    print(f"Config: {args.config}")
    print(f"Dataset: {summary['dataset']} (field '{summary['text_field']}')")
    print(f"Sequence length: {summary['seq_len']}")
    print(f"Tokens/step: {summary['tokens_per_step']:,}")
    print(f"Total tokens: {summary['total_tokens']:,}")
    print(f"Shards needed (@ {summary['tokens_per_shard']:,} tokens/shard): {summary['shards_needed']}")


def open_shard(out_dir: Path, idx: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shard_{idx:05d}.txt"
    return path, path.open("w", encoding="utf-8")


def stream_cmd(args):
    model_cfg, data_cfg, train_cfg = load_yaml_config(args.config)
    target_tokens = args.target_tokens or train_cfg.total_tokens(data_cfg.seq_len)
    tokenizer = SentencePieceTokenizer(args.tokenizer)

    auth = args.auth_token or os.environ.get("HF_TOKEN")
    ds = load_dataset(
        args.dataset or data_cfg.dataset,
        args.config_name or data_cfg.config,
        split=args.split or data_cfg.split,
        streaming=True,
        cache_dir=args.cache_dir,
        token=auth,
    )

    shard_tokens = 0
    shard_idx = 0
    total_tokens = 0
    total_rows = 0
    shard_path, shard_file = open_shard(args.output_dir, shard_idx)
    pbar = tqdm(total=target_tokens, unit="tok", desc="Streaming")

    for row in ds:
        text = row.get(args.text_field or data_cfg.text_field, "")
        if not isinstance(text, str) or not text.strip():
            continue

        if args.score_field and args.score_min is not None:
            try:
                if float(row.get(args.score_field, 0.0)) < args.score_min:
                    continue
            except Exception:
                continue

        text = text.replace("\n", " ").strip()
        if args.max_chars and len(text) > args.max_chars:
            text = text[: args.max_chars]
        if not text:
            continue

        ids = tokenizer.encode(text)
        tok_len = len(ids)
        if tok_len == 0:
            continue

        shard_file.write(text + "\n")
        shard_tokens += tok_len
        total_tokens += tok_len
        total_rows += 1
        pbar.update(tok_len)

        if shard_tokens >= args.tokens_per_shard:
            shard_file.close()
            shard_idx += 1
            shard_tokens = 0
            shard_path, shard_file = open_shard(args.output_dir, shard_idx)

        if total_tokens >= target_tokens:
            break

    shard_file.close()
    pbar.close()
    print(f"Wrote {total_rows} rows across {shard_idx + 1} shards -> {args.output_dir}")
    print(f"Approx tokens collected: {total_tokens:,}")


def _token_chunks_from_text(
    text_path: Path, tokenizer: SentencePieceTokenizer, seq_len: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    buffer: List[int] = []
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    with text_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            ids = tokenizer.encode(text)
            buffer.extend(ids + [tokenizer.eos_id])
            while len(buffer) >= seq_len + 1:
                chunk = buffer[: seq_len + 1]
                buffer = buffer[seq_len:]
                xs.append(np.array(chunk[:-1], dtype=np.uint16))
                ys.append(np.array(chunk[1:], dtype=np.uint16))
    if buffer and len(buffer) >= seq_len + 1:
        chunk = buffer[: seq_len + 1]
        xs.append(np.array(chunk[:-1], dtype=np.uint16))
        ys.append(np.array(chunk[1:], dtype=np.uint16))
    return xs, ys


def tokenize_cmd(args):
    tokenizer = SentencePieceTokenizer(args.tokenizer)
    input_shards = sorted(args.input_dir.glob("shard_*.txt"))
    if not input_shards:
        raise SystemExit(f"No text shards found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx, shard in enumerate(tqdm(input_shards, desc="Tokenizing")):
        xs, ys = _token_chunks_from_text(shard, tokenizer, args.seq_len)
        if not xs:
            continue
        x_arr = np.stack(xs)
        y_arr = np.stack(ys)
        if args.format == "npz":
            out_path = args.output_dir / f"shard_{idx:05d}.npz"
            np.savez(out_path, x=x_arr, y=y_arr)
        else:
            out_x = args.output_dir / f"shard_{idx:05d}_x.npy"
            out_y = args.output_dir / f"shard_{idx:05d}_y.npy"
            np.save(out_x, x_arr.astype(np.uint16))
            np.save(out_y, y_arr.astype(np.uint16))
    if args.cleanup_text:
        for shard in input_shards:
            shard.unlink()
        print("Raw text shards removed after tokenization.")


def cleanup_cmd(args):
    removed = []
    for target in (args.text_dir, args.token_dir):
        if target and target.exists():
            for f in target.glob("shard_*"):
                f.unlink()
            removed.append(str(target))
    if removed:
        print(f"Removed shard files in: {', '.join(removed)}")
    else:
        print("Nothing to cleanup.")


def build_parser():
    p = argparse.ArgumentParser(description="Unified pretrain data pipeline.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Report token budget and shard count.")
    p_plan.add_argument("--config", type=Path, required=True, help="Training YAML config.")
    p_plan.add_argument("--tokens-per-shard", type=int, default=50_000_000, help="Tokens per raw shard.")
    p_plan.set_defaults(func=plan_cmd)

    p_stream = sub.add_parser("stream", help="Stream dataset into raw text shards.")
    p_stream.add_argument("--config", type=Path, required=True, help="Training YAML config.")
    p_stream.add_argument("--dataset", type=str, help="HF dataset name (overrides config).")
    p_stream.add_argument("--config-name", type=str, help="HF dataset config name.")
    p_stream.add_argument("--split", type=str, help="Dataset split.")
    p_stream.add_argument("--text-field", type=str, help="Text field name.")
    p_stream.add_argument("--score-field", type=str, help="Optional numeric score field for filtering.")
    p_stream.add_argument("--score-min", type=float, help="Minimum score to keep a row.")
    p_stream.add_argument("--max-chars", type=int, default=12000, help="Trim text longer than this.")
    p_stream.add_argument("--tokenizer", type=Path, required=True, help="SentencePiece tokenizer model.")
    p_stream.add_argument("--target-tokens", type=int, help="Token target (defaults to config total tokens).")
    p_stream.add_argument("--tokens-per-shard", type=int, default=50_000_000, help="Tokens per text shard.")
    p_stream.add_argument("--output-dir", type=Path, required=True, help="Output directory for text shards.")
    p_stream.add_argument("--cache-dir", type=Path, help="HF cache directory.")
    p_stream.add_argument("--auth-token", type=str, help="HF auth token or HF_TOKEN env var.")
    p_stream.set_defaults(func=stream_cmd)

    p_tok = sub.add_parser("tokenize", help="Pretokenize text shards into npy/npz shards.")
    p_tok.add_argument("--input-dir", type=Path, required=True, help="Directory containing text shards.")
    p_tok.add_argument("--output-dir", type=Path, required=True, help="Directory for token shards.")
    p_tok.add_argument("--tokenizer", type=Path, required=True, help="Tokenizer model.")
    p_tok.add_argument("--seq-len", type=int, default=2048, help="Sequence length for chunks.")
    p_tok.add_argument("--format", choices=["npy", "npz"], default="npy", help="Shard format (npy preferred for mmap).")
    p_tok.add_argument("--cleanup-text", action="store_true", help="Delete text shards after tokenization.")
    p_tok.set_defaults(func=tokenize_cmd)

    p_clean = sub.add_parser("cleanup", help="Remove raw/tokenized shards.")
    p_clean.add_argument("--text-dir", type=Path, help="Text shard directory to clean.")
    p_clean.add_argument("--token-dir", type=Path, help="Pretokenized shard directory to clean.")
    p_clean.set_defaults(func=cleanup_cmd)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
