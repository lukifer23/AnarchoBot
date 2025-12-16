from pathlib import Path
from typing import Iterable, List, Tuple, Iterator
import pickle
import numpy as np

import mlx.core as mx

from anarchobot.tokenizer import SentencePieceTokenizer


class StreamingShardLoader:
    """
    Memory-efficient streaming loader for pre-tokenized shards.
    Uses memory mapping to avoid loading entire shards into RAM.
    """

    def __init__(self, shard_dir: Path, batch_size: int, seq_len: int, format: str = "npy"):
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.format = format

        if format == "npy":
            self.shard_files = sorted(self.shard_dir.glob("shard_*_x.npy"))
        elif format == "mlx":
            self.shard_files = sorted(self.shard_dir.glob("shard_*.mlx"))
        else:
            raise ValueError(f"Unsupported format: {format}")

        if not self.shard_files:
            raise FileNotFoundError(f"No shards found in {self.shard_dir}")

        self.current_shard_idx = 0
        self.current_pos = 0
        self.x_data = None
        self.y_data = None
        self._load_current_shard()

    def _load_current_shard(self):
        """Load current shard using memory mapping."""
        if self.format == "npy":
            x_file = self.shard_files[self.current_shard_idx]
            y_file = x_file.with_name(x_file.name.replace("_x.npy", "_y.npy"))

            # Get shapes first (fast metadata read)
            temp_x = np.load(str(x_file), mmap_mode='r')
            temp_y = np.load(str(y_file), mmap_mode='r')

            if temp_x.shape != temp_y.shape:
                raise ValueError(f"Shape mismatch: {x_file} vs {y_file}")

            # Create memory-mapped arrays for zero-copy access
            self.x_data = np.memmap(str(x_file), dtype=temp_x.dtype, mode='r', shape=temp_x.shape)
            self.y_data = np.memmap(str(y_file), dtype=temp_y.dtype, mode='r', shape=temp_y.shape)
        else:
            # For MLX format, load the entire shard (less optimal but compatible)
            with self.shard_files[self.current_shard_idx].open("rb") as f:
                data = pickle.load(f)
                self.x_data = data["x"]
                self.y_data = data["y"]

        self.current_pos = 0

    def __iter__(self) -> Iterator[Tuple[mx.array, mx.array]]:
        return self

    def __next__(self) -> Tuple[mx.array, mx.array]:
        if self.current_pos + self.batch_size > len(self.x_data):
            # Move to next shard
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_files)
            self._load_current_shard()

        start = self.current_pos
        end = start + self.batch_size

        # Extract batch and convert to MLX arrays; convert memmap slice to numpy first
        x_np = np.asarray(self.x_data[start:end])
        y_np = np.asarray(self.y_data[start:end])
        x_batch = mx.array(x_np, dtype=mx.int32)
        y_batch = mx.array(y_np, dtype=mx.int32)

        self.current_pos = end
        return x_batch, y_batch


def shard_paths(shard_dir: Path, extension: str = "mlx") -> List[Path]:
    """Find shards with specified extension."""
    if extension == "npy":
        # NPY format has separate x/y files, return x files only
        return sorted(Path(shard_dir).glob("shard_*_x.npy"))
    elif extension == "mlx":
        return sorted(Path(shard_dir).glob("shard_*.mlx"))
    elif extension == "txt":
        return sorted(Path(shard_dir).glob("shard_*.txt"))
    else:
        return sorted(Path(shard_dir).glob(f"shard_*.{extension}"))


def stream_mlx_shards(shards: List[Path]) -> Iterable[Tuple[mx.array, mx.array]]:
    """Stream pre-tokenized MLX shards."""
    for shard in shards:
        if shard.name.endswith('_x.npy'):
            # Handle NPY format (separate x/y files) with memory mapping
            x_file = shard
            y_file = shard.parent / shard.name.replace('_x.npy', '_y.npy')

            # Get shapes and dtypes first (can't use context manager with np.load)
            temp_x = np.load(str(x_file))
            x_shape = temp_x.shape
            x_dtype = temp_x.dtype
            del temp_x  # Free memory

            temp_y = np.load(str(y_file))
            y_shape = temp_y.shape
            y_dtype = temp_y.dtype
            del temp_y  # Free memory

            # Memory map the arrays for zero-copy loading
            x_data = np.memmap(str(x_file), dtype=x_dtype, mode='r', shape=x_shape)
            y_data = np.memmap(str(y_file), dtype=y_dtype, mode='r', shape=y_shape)

            # Stream each sequence
            for i in range(len(x_data)):
                yield mx.array(x_data[i], dtype=mx.int32), mx.array(y_data[i], dtype=mx.int32)

        elif shard.suffix == ".npz":
            data = np.load(shard)
            x_arr = data["x"]
            y_arr = data["y"]
            for i in range(len(x_arr)):
                yield mx.array(x_arr[i]), mx.array(y_arr[i])
        else:
            # Handle MLX format (pickle)
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
        format: "mlx" for pre-tokenized pickle, "npz" for pre-tokenized npz, "npy" for memmap npy pairs, "txt" for raw text
    """
    if format in ("mlx", "npz"):
        shards = shard_paths(shard_dir, extension=format)
        for x, y in stream_mlx_shards(shards):
            yield x, y
    elif format in ("npy", "mmap"):
        x_shards = sorted(Path(shard_dir).glob("shard_*_x.npy"))
        if not x_shards:
            raise FileNotFoundError(f"No npy shards found in {shard_dir}")
        for x_path in x_shards:
            y_path = x_path.with_name(x_path.name.replace("_x.npy", "_y.npy"))
            if not y_path.exists():
                raise FileNotFoundError(f"Missing y shard for {x_path}")

            # Use true memory mapping for zero-copy loading
            # First get the shape and dtype from the file
            temp_x = np.load(str(x_path))
            x_shape = temp_x.shape
            x_dtype = temp_x.dtype
            del temp_x  # Free memory

            temp_y = np.load(str(y_path))
            y_shape = temp_y.shape
            y_dtype = temp_y.dtype
            del temp_y  # Free memory

            # Create memory-mapped arrays
            x_arr = np.memmap(str(x_path), dtype=x_dtype, mode='r', shape=x_shape)
            y_arr = np.memmap(str(y_path), dtype=y_dtype, mode='r', shape=y_shape)

            if x_arr.shape != y_arr.shape:
                raise ValueError(f"Shape mismatch between {x_path} and {y_path}")

            # Stream sequences one by one (true memory mapping)
            for i in range(x_arr.shape[0]):
                x_np = np.asarray(x_arr[i])
                y_np = np.asarray(y_arr[i])
                yield mx.array(x_np, dtype=mx.int32), mx.array(y_np, dtype=mx.int32)
    elif format == "txt":
        if tokenizer is None:
            raise ValueError("tokenizer required for txt format")
        buffer: List[int] = []
        for text in stream_text_shards(shard_dir.glob("shard_*.txt")):
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
