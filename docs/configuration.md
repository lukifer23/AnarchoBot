# Configuration Guide

## Overview

AnarchoBot uses YAML configuration files to control all aspects of training. Configurations are split into three main sections: `model`, `data`, and `train`.

## Model Configuration

```yaml
model:
  vocab_size: 32000          # Must match tokenizer vocabulary
  n_layers: 12               # Transformer layers (6-24 typical)
  d_model: 768              # Hidden dimension (512-1024 typical)
  n_heads: 12               # Attention heads (d_model must be divisible by n_heads)
  mlp_multiple: 4.0         # MLP expansion factor (3.0-8.0 typical)
  dropout: 0.1              # Dropout rate (0.0-0.2)
  max_seq_len: 4096         # Maximum sequence length for positional encoding
  rope_theta: 10000.0       # RoPE base frequency
  ffn_activation: "silu"     # Activation function ("silu" or "gelu")
  norm_eps: 1e-5            # RMSNorm epsilon
  tie_embeddings: true      # Tie input/output embeddings
```

### Model Size Guidelines

| Model Size | Parameters | d_model | n_layers | n_heads | Memory (GB) |
|------------|------------|---------|----------|---------|-------------|
| Tiny | 7M | 256 | 6 | 8 | 0.5 |
| Small | 30M | 512 | 8 | 8 | 1.0 |
| Medium | 137M | 768 | 12 | 12 | 2.5 |
| Large | 500M | 1024 | 24 | 16 | 8.0 |

## Data Configuration

```yaml
data:
  dataset: "allenai/c4"           # HuggingFace dataset name
  config: "en"                   # Dataset config (if needed)
  split: "train"                 # Dataset split
  text_field: "text"             # Field containing text data
  seq_len: 2048                  # Training sequence length
  shuffle_buffer: 1000           # Streaming shuffle buffer size
  num_workers: 2                 # Data loading workers
  cache_dir: null                # HF cache directory
  streaming: true                # Use streaming for large datasets
```

### Dataset Options

#### Pre-training Datasets
- **C4**: `"allenai/c4"` (config: `"en"`)
- **Ultra-FineWeb**: `"EliMC/Ultra-FineWeb"` (text_field: `"content"`)
- **Pile Uncensored**: `"NeelNanda/pile-uncopyrighted"`
- **FineWeb**: `"HuggingFaceFW/fineweb"`

#### Fine-tuning Datasets
- **UltraChat**: `"HuggingFaceH4/ultrachat_200k"`
- **OpenAssistant**: `"OpenAssistant/oasst1"`

#### Preference Tuning Datasets
- **UltraFeedback**: `"HuggingFaceH4/ultrafeedback_binarized"`

## Training Configuration

```yaml
train:
  total_steps: 50000            # Total training steps
  micro_batch_size: 1           # Per-device batch size (keep at 1 for MPS)
  grad_accum_steps: 4           # Gradient accumulation steps
  lr: 3e-4                      # Peak learning rate
  min_lr: 3e-5                  # Minimum learning rate
  warmup_steps: 1000            # Learning rate warmup steps
  weight_decay: 0.02            # Weight decay (decoupled)
  max_grad_norm: 1.0            # Gradient clipping threshold
  log_interval: 25              # Logging interval (steps)
  eval_interval: 500            # Evaluation interval (steps)
  ckpt_interval: 2000           # Checkpoint interval (steps)
  ckpt_keep: 3                  # Number of checkpoints to keep
  log_path: null                # Custom log directory
  save_dir: "checkpoints/pretrain"  # Checkpoint save directory
  precision: "float16"          # Precision ("float16" or "float32")
  compile: false                # PyTorch compilation (experimental)
  gradient_checkpointing: true  # Memory saving technique
  tokenizer_path: "data/tokenizer.model"  # Tokenizer path
  checkpoint_path: null         # Resume from checkpoint
```

## Advanced Configuration

### Memory Optimization

For M3 Pro 18GB, use these settings:

```yaml
# Minimal memory usage
micro_batch_size: 1
grad_accum_steps: 1
seq_len: 1024
gradient_checkpointing: true
precision: "float16"

# Higher throughput (moderate memory)
micro_batch_size: 1
grad_accum_steps: 4
seq_len: 2048
gradient_checkpointing: true
precision: "float16"
```

### Learning Rate Scheduling

```yaml
# Standard schedule
lr: 3e-4
min_lr: 3e-5
warmup_steps: 1000

# Smaller models (higher LR)
lr: 1e-3
min_lr: 1e-4
warmup_steps: 500

# Larger models (lower LR)
lr: 1e-4
min_lr: 1e-5
warmup_steps: 2000
```

### Optimizer Tuning

```yaml
# Muon + AdamW hybrid (default)
lr: 3e-4          # Muon learning rate
weight_decay: 0.02

# Pure AdamW (alternative)
# Set lr to AdamW rate, Muon parameters ignored

# Muon momentum tuning
# momentum: 0.95 (default, stable)
# momentum: 0.90 (more aggressive)
# momentum: 0.97 (more conservative)
```

## Configuration Validation

Use the setup verification script to check your configuration:

```bash
python scripts/setup_verification.py
```

This will:
- Verify all required fields are present
- Check model/data compatibility
- Estimate memory requirements
- Validate dataset access

## Example Configurations

### Minimal Test Run
```yaml
model:
  vocab_size: 32000
  n_layers: 6
  d_model: 512
  n_heads: 8

data:
  dataset: "allenai/c4"
  config: "en"
  seq_len: 1024

train:
  total_steps: 100
  grad_accum_steps: 1
  lr: 1e-3
  warmup_steps: 10
  ckpt_interval: 50
```

### Full Pre-training
```yaml
model:
  vocab_size: 32000
  n_layers: 12
  d_model: 768
  n_heads: 12

data:
  dataset: "EliMC/Ultra-FineWeb"
  text_field: "content"
  seq_len: 2048

train:
  total_steps: 50000
  grad_accum_steps: 4
  lr: 3e-4
  warmup_steps: 1000
  ckpt_interval: 2000
  ckpt_keep: 3
```

### Memory-Constrained Training
```yaml
model:
  vocab_size: 32000
  n_layers: 8
  d_model: 512
  n_heads: 8

data:
  seq_len: 1024

train:
  micro_batch_size: 1
  grad_accum_steps: 2
  gradient_checkpointing: true
  precision: "float16"
```

## Configuration Tips

### Scaling Laws
- **Model size**: Double parameters → halve learning rate
- **Batch size**: Larger batches → more stable training
- **Sequence length**: Longer sequences → lower learning rates
- **Dataset size**: More data → higher learning rates

### Backend-Specific Configuration

#### PyTorch MPS Configuration
- Automatic mixed precision support
- CUDA-style gradient scaling
- Compatible with existing PyTorch debugging tools

#### MLX Configuration
- Native Apple Silicon optimization
- Different data loading requirements (pre-tokenized shards)
- Alternative checkpoint format (.npz files)
- Higher throughput potential with proper data preparation

### Hardware-Specific Tuning

**M3 Pro 18GB:**
- **PyTorch MPS**: Keep `micro_batch_size: 1`, use `grad_accum_steps` for effective batch size
- **MLX**: Can use larger batches, optimized memory usage
- Enable `gradient_checkpointing` for seq_len > 1024 on both backends
- Monitor memory with `memory_monitor.py`

**M3 Max/Ultra:**
- Can use `micro_batch_size: 2`
- Higher `grad_accum_steps` possible
- Can disable `gradient_checkpointing` for shorter sequences

### Monitoring Configuration

Training automatically creates:
- TensorBoard logs: `save_dir/logs/tensorboard/`
- JSON metrics: `save_dir/logs/*.jsonl`
- Training summaries: `save_dir/logs/final_training_summary.json`

View training progress:
```bash
tensorboard --logdir checkpoints/pretrain/logs/tensorboard
```

## Troubleshooting

### Common Configuration Errors

**"size mismatch between tensors"**
- Check `vocab_size` matches tokenizer
- Ensure `d_model` divisible by `n_heads`

**"out of memory"**
- Reduce `seq_len` or `grad_accum_steps`
- Enable `gradient_checkpointing`

**"loss not decreasing"**
- Increase `lr` or `warmup_steps`
- Check data quality and tokenization

### Validation Checklist

Before starting training:
- [ ] `python scripts/setup_verification.py` passes
- [ ] Tokenizer exists and matches `vocab_size`
- [ ] Dataset accessible and contains expected fields
- [ ] Memory estimate looks reasonable
- [ ] Checkpoint directory is writable
- [ ] Sufficient disk space (checkpoints are ~500MB each)
