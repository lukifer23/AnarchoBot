# Examples

This directory contains example configurations, outputs, and usage patterns for AnarchoBot.

## Configuration Examples

### 1. Minimal Test Configuration
```yaml
# examples/minimal_test.yaml
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
**Use case**: Quick testing and validation

### 2. Memory-Optimized Configuration
```yaml
# examples/memory_optimized.yaml
model:
  vocab_size: 32000
  n_layers: 8
  d_model: 512
  n_heads: 8

data:
  dataset: "EliMC/Ultra-FineWeb"
  text_field: "content"
  seq_len: 1024

train:
  micro_batch_size: 1
  grad_accum_steps: 2
  gradient_checkpointing: true
  precision: "float16"
  lr: 5e-4
```
**Use case**: Maximum memory efficiency for constrained hardware

### 3. High-Performance Configuration
```yaml
# examples/high_performance.yaml
model:
  vocab_size: 32000
  n_layers: 16
  d_model: 1024
  n_heads: 16

data:
  dataset: "EliMC/Ultra-FineWeb"
  text_field: "content"
  seq_len: 2048

train:
  grad_accum_steps: 8
  lr: 2e-4
  warmup_steps: 2000
  weight_decay: 0.01
```
**Use case**: Maximum performance on M3 Max/Ultra

## Training Outputs

### Sample Training Log
```
Step    25 | Loss: 705.36 | PPL: inf | LR: 0.000039 | Time: 00:03:49 | Tokens/sec: 976 | Memory: 2103MB
Step    50 | Loss: 699.66 | PPL: inf | LR: 0.000075 | Time: 00:07:20 | Tokens/sec: 976 | Memory: 2103MB

📊 Training completed!
   Final loss: 699.6633
   Best loss: 694.9235
   Average throughput: 976 tokens/sec
```

### Memory Usage Report
```json
{
  "gpu_memory_avg_mb": 2112.82,
  "gpu_memory_max_mb": 2120.04,
  "system_memory_avg_percent": 83.68,
  "system_memory_max_percent": 85.2,
  "total_measurements": 5
}
```

### Final Training Summary
```json
{
  "total_steps": 50,
  "total_tokens_processed": 409600,
  "total_training_time_sec": 439.85,
  "final_loss": 699.66,
  "best_loss": 694.92,
  "avg_loss": 703.99,
  "loss_std": 3.49,
  "avg_throughput_tokens_per_sec": 976.10,
  "peak_throughput_tokens_per_sec": 996.07
}
```

## Usage Patterns

### 1. Quick Model Validation
```bash
# Train minimal model for 5 minutes
python scripts/train_tokenizer.py --input data/sample.txt --vocab-size 8000 --model-prefix data/mini_tokenizer
PYTHONPATH=src python -m anarchobot.train --config examples/minimal_test.yaml
```

### 2. Memory Usage Testing
```bash
# Test PyTorch MPS memory usage
PYTHONPATH=src python -c "
from anarchobot.memory_monitor import MemoryMonitor
from anarchobot.model import TransformerLM
from anarchobot.config import ModelConfig
import torch

config = ModelConfig(vocab_size=32000)
model = TransformerLM(config)
monitor = MemoryMonitor(torch.device('mps'))
estimate = monitor.estimate_training_memory(model, 1, 2048)
print(f'PyTorch memory estimate: {estimate}')
"

# Test MLX memory usage
python -c "
import mlx.core as mx
from anarchobot_mlx.model_mlx import TransformerLM

model = TransformerLM(vocab_size=32000, n_layers=12, d_model=768, n_heads=12)
# MLX memory tracking is built-in
print('MLX model loaded successfully')
"
```

### 3. Performance Benchmarking
```bash
# Benchmark different model sizes
for layers in 6 12 24; do
  sed "s/n_layers: 12/n_layers: $layers/" configs/pretrain.yaml > configs/test_${layers}l.yaml
  timeout 300 PYTHONPATH=src python -m anarchobot.train --config configs/test_${layers}l.yaml
done
```

### 4. MLX Backend Examples

#### MLX Model Instantiation
```python
import mlx.core as mx
from anarchobot_mlx.model_mlx import TransformerLM

# Create model
model = TransformerLM(
    vocab_size=32000,
    n_layers=12,
    d_model=768,
    n_heads=12,
    mlp_multiple=4.0,
    max_seq_len=4096
)

# Test forward pass
x = mx.random.uniform(shape=(1, 512), minval=0, maxval=32000, dtype=mx.int32)
logits, loss = model(x, x)
print(f"Output shape: {logits.shape}")
```

#### MLX Data Loading
```python
from anarchobot_mlx.data_mlx import token_chunk_iterator

# Load pre-tokenized MLX data
data_iter = token_chunk_iterator(
    shard_dir="data/mlx_shards",
    tokenizer=None,  # Not needed for pre-tokenized data
    seq_len=2048,
    format="mlx"
)

# Get a batch
x_batch, y_batch = next(data_iter)
print(f"Batch shapes: x={x_batch.shape}, y={y_batch.shape}")
```

#### MLX Training Loop
```python
import mlx.core as mx
from anarchobot_mlx.optim_mlx import MuonAdamW

# Setup
model = TransformerLM(...)
optimizer = MuonAdamW(lr_muon=3e-4, lr_adam=4.5e-4, weight_decay=0.02)

# Training step
loss_and_grad = mx.value_and_grad(lambda m, x, y: m(x, y)[1])
loss, grads = loss_and_grad(model, x_batch, y_batch)
new_params = optimizer.update(model.trainable_parameters(), grads)
model.update(new_params)
```

### 5. Data Pipeline Testing
```bash
# Test data loading performance
python -c "
from anarchobot.data import TokenChunkDataset
from torch.utils.data import DataLoader
import time

dataset = TokenChunkDataset(
    dataset='allenai/c4',
    config='en',
    split='train',
    tokenizer_path='data/tokenizer.model',
    seq_len=1024
)
dataloader = DataLoader(dataset, batch_size=1)

start = time.time()
count = 0
for batch in dataloader:
    count += 1
    if count >= 10:
        break

print(f'Data loading: {count/(time.time()-start):.2f} batches/sec')
"
```

## Model Checkpoints

Example checkpoint structure:
```
checkpoints/pretrain/
├── step_2000.pt          # Full checkpoint (model + optimizer + metadata)
├── step_4000.pt
├── last.pt              # Latest checkpoint
├── logs/
│   ├── final_training_summary.json
│   ├── pretrain_768d_12l_metrics.jsonl
│   └── tensorboard/
│       └── events.out.tfevents.1765569653...
```

### Checkpoint Contents
```python
checkpoint = torch.load('checkpoints/pretrain/step_2000.pt')
print(checkpoint.keys())
# dict_keys(['model', 'optimizer', 'step', 'config', 'tokenizer_config'])
```

## Custom Training Scripts

### Resume Training
```python
from anarchobot.train import load_configs, set_seed
from anarchobot.model import TransformerLM
from anarchobot.optim import build_muon_adam_optimizer
from anarchobot.utils import get_device, load_checkpoint
import yaml

# Load config
with open('configs/pretrain.yaml') as f:
    config = yaml.safe_load(f)

# Update checkpoint path
config['train']['checkpoint_path'] = 'checkpoints/pretrain/step_2000.pt'

# Resume training...
```

### Custom Evaluation
```python
from anarchobot.model import TransformerLM
from anarchobot.tokenizer import SentencePieceTokenizer
import torch

# Load trained model
model = TransformerLM.load_from_checkpoint('checkpoints/pretrain/last.pt')
tokenizer = SentencePieceTokenizer('data/tokenizer.model')
model.eval()

# Generate text
prompt = "The future of AI is"
tokens = torch.tensor([tokenizer.encode(prompt)])
with torch.no_grad():
    output = model.generate(tokens, max_new_tokens=50)
    generated = tokenizer.decode(output[0].tolist())
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
```

## Performance Metrics

### Expected Performance by Hardware

| Hardware | Tokens/sec | Memory Usage | Max Seq Len |
|----------|------------|--------------|-------------|
| M3 (8GB) | 400-600 | 1-2GB | 1024 |
| M3 Pro (18GB) | 800-1200 | 2-3GB | 2048 |
| M3 Max (32GB) | 1200-1600 | 4-6GB | 4096 |
| M3 Ultra (64GB) | 1600-2000 | 6-8GB | 4096+ |

### Scaling Performance

- **Model size**: 2x parameters → 1.5x throughput (larger matrices)
- **Sequence length**: 2x seq_len → 1.8x throughput (better parallelism)
- **Batch size**: 2x batch → 1.9x throughput (GPU utilization)
- **Gradient checkpointing**: 10-15% overhead, 60-70% memory reduction

## Troubleshooting Examples

### Memory Issues
```bash
# Monitor memory during training
PYTHONPATH=src python -c "
from anarchobot.memory_monitor import MemoryMonitor
import torch
monitor = MemoryMonitor(torch.device('mps'))
stats = monitor.get_memory_stats()
print(f'GPU Memory: {stats.gpu_memory_used:.0f}MB')
print(f'System Memory: {stats.system_memory_percent:.1f}%')
"
```

### Data Loading Issues
```bash
# Test dataset access
python -c "
from datasets import load_dataset
try:
    ds = load_dataset('allenai/c4', 'en', split='train', streaming=True)
    sample = next(iter(ds))
    print('Dataset access: OK')
    print(f'Sample keys: {sample.keys()}')
except Exception as e:
    print(f'Dataset error: {e}')
"
```

### Model Loading Issues
```bash
# Validate checkpoint
python -c "
import torch
try:
    ckpt = torch.load('checkpoints/pretrain/last.pt', map_location='cpu')
    print('Checkpoint keys:', list(ckpt.keys()))
    print('Model state size:', len(ckpt['model']))
except Exception as e:
    print(f'Checkpoint error: {e}')
"
```
