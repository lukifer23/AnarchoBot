# Quick Start Guide

## Prerequisites

- **Hardware**: Apple Silicon Mac (M3/M4) with at least 18GB RAM
- **Software**: macOS with PyTorch MPS support
- **Python**: 3.10+ with pip

## Installation (5 minutes)

```bash
# Clone the repository
git clone https://github.com/lukifer23/AnarchoBot.git
cd AnarchoBot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Verify setup
python scripts/setup_verification.py
```

## Test Training (10 minutes)

```bash
# Download a small dataset sample
python scripts/stream_text_dataset.py \
  --dataset EliMC/Ultra-FineWeb \
  --split en \
  --text-field content \
  --samples 200000 \
  --score-field score \
  --score-min 0.8 \
  --output data/ultrafineweb_sample.txt

# Train a basic tokenizer
python scripts/train_tokenizer.py \
  --input data/ultrafineweb_sample.txt \
  --vocab-size 32000 \
  --model-prefix data/tokenizer

# Run a test training session
PYTHONPATH=src python -m anarchobot.train \
  --config configs/pretrain.yaml
```

## Expected Output

```
Using device: mps
🔍 Checking memory requirements...
✅ PASS: Memory estimate: 526MB params, ~2103MB training memory
✅ PASS: All dependencies available

Step    25 | Loss: 705.36 | PPL: inf | LR: 0.000039 | Time: 00:03:49 | Tokens/sec: 976 | Memory: 2103MB
📊 Training completed!
   Final memory usage: {'gpu_memory_avg_mb': 2112.82, 'gpu_memory_max_mb': 2120.04}
```

## Full Training Pipeline

### 1. Large-Scale Data Preparation (2-4 hours)
```bash
# Download 2.8B tokens of high-quality training data
python scripts/stream_for_token_budget.py \
  --dataset EliMC/Ultra-FineWeb \
  --split en \
  --text-field content \
  --tokenizer data/tokenizer.model \
  --target-tokens 2800000000 \
  --tokens-per-shard 50000000 \
  --score-field score \
  --score-min 0.8 \
  --output-dir data/ultrafineweb_full
```

### 2. Improved Tokenizer Training (10 minutes)
```bash
# Train tokenizer on 30M+ tokens for better quality
python scripts/train_tokenizer.py \
  --input data/ultrafineweb_full/shard_00000.txt \
  --vocab-size 32000 \
  --model-prefix data/tokenizer_v2
```

### 3. Large-Scale Pre-training (30-60 days)
```bash
# Update config to use new tokenizer and full dataset
# Edit configs/pretrain.yaml to point to data/tokenizer_v2.model
# and data/ultrafineweb_full/ directory

PYTHONPATH=src python -m anarchobot.train \
  --config configs/pretrain.yaml
```

### 4. Supervised Fine-tuning (Optional)
```bash
PYTHONPATH=src python -m anarchobot.finetune \
  --config configs/sft.yaml \
  --base-checkpoint checkpoints/pretrain/last.pt
```

### 5. Chat Interface
```bash
python scripts/chat.py \
  --config configs/sft.yaml \
  --checkpoint checkpoints/sft/last.pt
```

## Monitoring Training Progress

Training logs are automatically saved to:
- **TensorBoard**: `checkpoints/pretrain/logs/tensorboard/`
- **JSON Metrics**: `checkpoints/pretrain/logs/*.jsonl`
- **Training Summary**: `checkpoints/pretrain/logs/final_training_summary.json`

View TensorBoard:
```bash
pip install tensorboard
tensorboard --logdir checkpoints/pretrain/logs/tensorboard
```

## Configuration Tuning

### Memory Optimization
- Keep `micro_batch_size: 1`
- Adjust `grad_accum_steps` to control effective batch size
- Enable `gradient_checkpointing: true` for long contexts

### Performance Tuning
- Increase `seq_len` gradually during training
- Adjust `lr` based on model size (smaller models need higher LR)
- Monitor memory usage with `memory_monitor.py`

### Model Scaling
Use `scripts/model_stats.py` to calculate optimal model size:
```bash
python scripts/model_stats.py --config configs/pretrain.yaml
```

## Hardware Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Chip | M3 | M3 Pro/Max/Ultra |
| RAM | 16GB | 18GB+ |
| Storage | 50GB | 200GB+ |
| OS | macOS 13+ | macOS 14+ |

## Common Issues

### MPS Not Available
```
RuntimeError: MPS not available
```
**Solution**: Ensure PyTorch is installed with MPS support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Out of Memory
```
RuntimeError: MPS out of memory
```
**Solutions**:
- Reduce `micro_batch_size` to 1
- Increase `grad_accum_steps`
- Enable `gradient_checkpointing`
- Reduce `seq_len`

### Slow Training
- Ensure PyTorch MPS is using the GPU: `torch.mps.current_allocated_memory() > 0`
- Check Activity Monitor for GPU utilization
- Try different `lr` values

## Getting Help

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check `docs/` directory for detailed guides
- **Training Logs**: Include `final_training_summary.json` when reporting issues
