from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset

from .tokenizer import SentencePieceTokenizer


def _format_messages(messages: List[dict]) -> str:
    """
    Format a list of {"role": str, "content": str} messages into a plain text chat transcript.
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


class TokenChunkDataset(IterableDataset):
    """
    Iterable dataset that streams text, tokenizes, and yields fixed-length token chunks.
    """

    def __init__(
        self,
        dataset: str,
        split: str,
        text_field: str,
        tokenizer: SentencePieceTokenizer,
        seq_len: int,
        shuffle_buffer: int = 1000,
        streaming: bool = True,
        cache_dir=None,
        config: str = None,
    ):
        super().__init__()
        self.dataset_name = dataset
        self.split = split
        self.text_field = text_field
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer
        self.streaming = streaming
        self.cache_dir = cache_dir
        self.config = config

        if config:
            ds = load_dataset(dataset, config, split=split, streaming=streaming, cache_dir=cache_dir)
        else:
            ds = load_dataset(dataset, split=split, streaming=streaming, cache_dir=cache_dir)
        self.ds = ds.shuffle(buffer_size=shuffle_buffer) if streaming and shuffle_buffer else ds

    def _extract_text(self, sample) -> str:
        if self.text_field in sample:
            value = sample[self.text_field]
            if isinstance(value, list):
                if value and isinstance(value[0], dict):
                    return _format_messages(value)
                return "\n".join(str(v) for v in value)
            return str(value)
        if "messages" in sample:
            return _format_messages(sample["messages"])
        if "conversation" in sample:
            return _format_messages(sample["conversation"])
        # fall back to string conversion
        return str(sample)

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        buffer: List[int] = []
        for sample in self.ds:
            text = self._extract_text(sample)
            if not text:
                continue
            ids = self.tokenizer.encode(text)
            buffer.extend(ids + [self.tokenizer.eos_id])
            while len(buffer) > self.seq_len:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


class MemmapShardDataset(IterableDataset):
    """
    Iterable dataset that streams pre-tokenized shards stored as memory-mapped numpy arrays.

    Expects pairs of files: shard_XXXXX_x.npy and shard_XXXXX_y.npy (dtype uint16 or int32).
    """

    def __init__(self, shard_dir: str, seq_len: int, fmt: str = "npy"):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.seq_len = seq_len
        self.fmt = fmt
        if fmt not in ("npy", "mmap"):
            raise ValueError(f"Unsupported memmap format: {fmt}")
        self.shards = sorted(self.shard_dir.glob("shard_*_x.npy"))
        if not self.shards:
            raise FileNotFoundError(f"No shards found in {self.shard_dir} (expected shard_*_x.npy)")

    def __iter__(self):
        import numpy as np

        for shard_x in self.shards:
            shard_y = shard_x.with_name(shard_x.name.replace("_x.npy", "_y.npy"))
            x_arr = np.load(shard_x, mmap_mode="r")
            y_arr = np.load(shard_y, mmap_mode="r")
            if x_arr.shape != y_arr.shape:
                raise ValueError(f"Shape mismatch between {shard_x} and {shard_y}")
            for i in range(x_arr.shape[0]):
                yield torch.from_numpy(x_arr[i].astype("int64")), torch.from_numpy(y_arr[i].astype("int64"))
