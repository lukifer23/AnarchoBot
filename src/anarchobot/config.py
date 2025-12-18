from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import yaml


@dataclass
class ModelConfig:
    vocab_size: int
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    mlp_multiple: float = 4.0
    dropout: float = 0.1
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    ffn_activation: str = "silu"
    norm_eps: float = 1e-5
    tie_embeddings: bool = True


@dataclass
class DataConfig:
    dataset: str = "allenai/c4"
    config: Optional[str] = None
    split: str = "train"
    text_field: str = "text"
    seq_len: int = 2048
    shuffle_buffer: int = 1000
    num_workers: int = 2
    cache_dir: Optional[Path] = None
    streaming: bool = True
    shard_dir: Optional[Path] = None  # Optional pretokenized shard directory
    format: str = "stream"  # "stream" (default) or "npy" for memmap shards


@dataclass
class TrainingConfig:
    total_steps: int = 50000
    micro_batch_size: int = 1
    grad_accum_steps: int = 4
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.02
    max_grad_norm: float = 1.0
    log_interval: int = 25
    eval_interval: int = 500
    ckpt_interval: int = 2000
    ckpt_keep: int = 3
    save_dir: Path = Path("checkpoints")
    log_path: Optional[Path] = None
    precision: str = "float16"  # "float16" or "bfloat16"
    compile: bool = False
    gradient_checkpointing: bool = True
    tokenizer_path: Path = Path("data/tokenizer.model")
    checkpoint_path: Optional[Path] = None
    optimizer: str = "muon_adam"  # muon_adam | adamw
    adam_lr_multiplier: float = 1.5

    def tokens_per_step(self, seq_len: int) -> int:
        return self.micro_batch_size * self.grad_accum_steps * seq_len

    def total_tokens(self, seq_len: int) -> int:
        return self.tokens_per_step(seq_len) * self.total_steps

    def as_logging_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (int, float, bool, str)):
                out[k] = v
            else:
                out[k] = str(v)
        return out


def _coerce_path(value):
    if value is None or value == "":
        return None
    return Path(value)


def load_yaml_config(path: Path) -> Tuple[ModelConfig, DataConfig, TrainingConfig]:
    cfg = yaml.safe_load(path.read_text())

    model_dict = dict(cfg.get("model", {}))
    model_cfg = ModelConfig(**model_dict)

    data_dict = dict(cfg.get("data", {}))
    if data_dict.get("cache_dir") is not None:
        data_dict["cache_dir"] = _coerce_path(data_dict["cache_dir"])
    if data_dict.get("shard_dir") is not None:
        data_dict["shard_dir"] = _coerce_path(data_dict["shard_dir"])
    data_cfg = DataConfig(**data_dict)

    train_dict = dict(cfg.get("train", {}))
    for key in ["save_dir", "tokenizer_path", "checkpoint_path", "log_path"]:
        if train_dict.get(key) is not None:
            train_dict[key] = _coerce_path(train_dict[key])
    train_cfg = TrainingConfig(**train_dict)

    return model_cfg, data_cfg, train_cfg
