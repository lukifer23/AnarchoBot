from pathlib import Path
from typing import List

import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, model_file: Path):
        self.model_file = Path(model_file)
        self.sp = spm.SentencePieceProcessor(model_file=str(model_file))

    @classmethod
    def train(
        cls,
        input_paths: List[Path],
        vocab_size: int = 32000,
        model_prefix: Path = Path("data/tokenizer"),
        character_coverage: float = 0.9995,
        model_type: str = "bpe",
    ) -> "SentencePieceTokenizer":
        """
        Train a sentencepiece tokenizer on a set of text files.
        """
        input_arg = ",".join(str(p) for p in input_paths)
        model_prefix = Path(model_prefix)
        model_prefix.parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.Train(
            input=input_arg,
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
        return cls(model_prefix.with_suffix(".model"))

    def encode(self, text: str) -> List[int]:
        return list(self.sp.encode(text, out_type=int))

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

    @property
    def pad_id(self) -> int:
        return int(self.sp.pad_id())

    @property
    def eos_id(self) -> int:
        return int(self.sp.eos_id())

    @property
    def bos_id(self) -> int:
        return int(self.sp.bos_id())

    @property
    def vocab_size(self) -> int:
        return int(self.sp.vocab_size())
