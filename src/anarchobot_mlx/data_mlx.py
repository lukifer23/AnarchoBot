from pathlib import Path
from typing import Iterable, List, Tuple
import pickle
import numpy as np

import mlx.core as mx

from anarchobot.tokenizer import SentencePieceTokenizer


def shard_paths(shard_dir: Path, extension: str = "mlx") -> List[Path]:
    """Find shards with specified extension."""
    suffix = "mlx" if extension == "mlx" else extension
    return sorted(Path(shard_dir).glob(f"shard_*.{suffix}"))


def stream_mlx_shards(shards: List[Path]) -> Iterable[Tuple[mx.array, mx.array]]:
    """Stream pre-tokenized MLX shards."""
    for shard in shards:
        if shard.suffix == ".npz":
            data = np.load(shard)
            x_arr = data["x"]
            y_arr = data["y"]
            for i in range(len(x_arr)):
                yield mx.array(x_arr[i]), mx.array(y_arr[i])
        else:
            with shard.open("rb") as f:
                try:
                    while True:
                        data = pickle.load(f)
                        yield data["x"], data["y"]
                except EOFError:
                    continue


def stream_text_shards(shards: List[Path]) -> Iterable[str]:
    """Stream raw text shards (for compatibility)."""
    for shard in shards:
        with shard.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    yield text


def token_chunk_iterator(
    shard_dir: Path,
    tokenizer: SentencePieceTokenizer = None,
    seq_len: int = 2048,
    format: str = "mlx"
) -> Iterable[Tuple[mx.array, mx.array]]:
    """
    Iterator for training data.

    Args:
        shard_dir: Directory containing shards
        tokenizer: SentencePiece tokenizer (required for text format)
        seq_len: Sequence length
        format: "mlx" for pre-tokenized, "txt" for raw text
    """
    shards = shard_paths(shard_dir, extension=format)

    if format == "mlx":
        # Pre-tokenized format (optimal)
        for x, y in stream_mlx_shards(shards):
            yield x, y
    elif format == "txt":
        # Raw text format (slower, for compatibility)
        if tokenizer is None:
            raise ValueError("tokenizer required for txt format")
        buffer: List[int] = []
        for text in stream_text_shards(shards):
            ids = tokenizer.encode(text)
            buffer.extend(ids + [tokenizer.eos_id])
            while len(buffer) > seq_len:
                chunk = buffer[: seq_len + 1]
                buffer = buffer[seq_len:]
                x = mx.array(chunk[:-1], dtype=mx.int32)
                y = mx.array(chunk[1:], dtype=mx.int32)
                yield x, y
    else:
        raise ValueError(f"Unsupported format: {format}")
