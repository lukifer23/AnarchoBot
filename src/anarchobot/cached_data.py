"""
Cached data loader for improved MPS performance.
Pre-tokenizes and caches sequences to disk for faster loading.
"""
import pickle
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
import torch
from torch.utils.data import IterableDataset

from .tokenizer import SentencePieceTokenizer


class CachedTokenDataset(IterableDataset):
    """
    Cached dataset that pre-tokenizes data and stores to disk.
    Much faster than on-the-fly tokenization for repeated access.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        text_field: str,
        tokenizer: SentencePieceTokenizer,
        seq_len: int,
        cache_dir: Path,
        max_samples: Optional[int] = None,
        force_rebuild: bool = False,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.text_field = text_field
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.cache_dir = Path(cache_dir)
        self.max_samples = max_samples

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{dataset_name}_{split}_{seq_len}.pkl"

        # Build cache if needed
        if force_rebuild or not self.cache_file.exists():
            self._build_cache()
        else:
            print(f"Using cached data: {self.cache_file}")

    def _build_cache(self):
        """Build and save tokenized data cache."""
        print(f"Building cache for {self.dataset_name} {self.split}...")
        from datasets import load_dataset

        # Load dataset
        ds = load_dataset(self.dataset_name, split=self.split, streaming=True)

        sequences = []
        buffer: List[int] = []
        sample_count = 0

        for sample in ds:
            if self.max_samples and sample_count >= self.max_samples:
                break

            text = self._extract_text(sample)
            if not text:
                continue

            # Tokenize and add to buffer
            ids = self.tokenizer.encode(text)
            buffer.extend(ids + [self.tokenizer.eos_id])

            # Extract complete sequences
            while len(buffer) > self.seq_len:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len:]

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                sequences.append((x, y))

            sample_count += 1
            if sample_count % 1000 == 0:
                print(f"Processed {sample_count} samples, {len(sequences)} sequences")

        # Save to disk
        print(f"Saving {len(sequences)} sequences to {self.cache_file}")
        with open(self.cache_file, 'wb') as f:
            pickle.dump(sequences, f)

        self.sequences = sequences
        print(f"Cache built: {len(sequences)} sequences")

    def _extract_text(self, sample) -> str:
        """Extract text from sample (same logic as original)."""
        if self.text_field in sample:
            value = sample[self.text_field]
            if isinstance(value, list):
                if value and isinstance(value[0], dict):
                    return self._format_messages(value)
                return "\n".join(str(v) for v in value)
            return str(value)
        if "messages" in sample:
            return self._format_messages(sample["messages"])
        if "conversation" in sample:
            return self._format_messages(sample["conversation"])
        return str(sample)

    def _format_messages(self, messages: List[dict]) -> str:
        """Format chat messages."""
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """Load from cache and yield sequences."""
        if not hasattr(self, 'sequences'):
            print(f"Loading cached data from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.sequences = pickle.load(f)
            print(f"Loaded {len(self.sequences)} sequences")

        for x, y in self.sequences:
            yield x.clone(), y.clone()  # Clone to avoid shared memory issues


def create_efficient_data_loader(
    dataset_name: str,
    split: str,
    text_field: str,
    tokenizer: SentencePieceTokenizer,
    seq_len: int,
    batch_size: int,
    cache_dir: Path,
    num_workers: int = 2,
    max_samples: Optional[int] = None,
    force_rebuild_cache: bool = False,
):
    """
    Create an optimized data loader with caching.

    Returns a DataLoader with cached tokenized data for much faster iteration.
    """
    dataset = CachedTokenDataset(
        dataset_name=dataset_name,
        split=split,
        text_field=text_field,
        tokenizer=tokenizer,
        seq_len=seq_len,
        cache_dir=cache_dir,
        max_samples=max_samples,
        force_rebuild=force_rebuild_cache,
    )

    # Create DataLoader with optimized settings for MPS
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Streaming data doesn't need shuffle
        num_workers=num_workers,
        pin_memory=False,  # MPS doesn't benefit from pin_memory
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    return data_loader
