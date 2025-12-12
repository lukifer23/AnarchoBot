from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
