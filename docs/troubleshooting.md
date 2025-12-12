# Troubleshooting Guide

## Common Issues and Solutions

### 1. MPS Backend Issues

#### "MPS not available. Ensure you are on Apple Silicon with PyTorch built with MPS"
```
Error: RuntimeError: MPS not available
```

**Solutions:**
```bash
# Check if MPS is available
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Reinstall PyTorch with MPS support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Verify installation
python -c "import torch; print('MPS available:', torch.backends.mps.is_available()); print('MPS built:', torch.backends.mps.is_built())"
```

#### Training runs on CPU instead of GPU
```bash
# Check MPS memory usage
python -c "import torch; torch.mps.current_allocated_memory(); print('MPS memory:', torch.mps.current_allocated_memory())"
```

**Solutions:**
- Ensure PyTorch MPS is properly installed
- Check Activity Monitor → GPU tab for utilization
- Try restarting Python session

### 2. Memory Issues

#### "MPS out of memory" during training
```
RuntimeError: MPS out of memory during forward pass
```

**Immediate fixes (in config):**
```yaml
# Reduce memory usage
micro_batch_size: 1  # Keep at 1
grad_accum_steps: 1  # Start with 1, increase gradually
seq_len: 1024        # Reduce from 2048
gradient_checkpointing: true  # Enable memory saving
```

**Progressive optimization:**
1. `grad_accum_steps: 1` - minimal memory
2. `seq_len: 512` - shortest sequences
3. `gradient_checkpointing: true` - 60-70% memory reduction
4. `precision: float32` - higher precision but more stable

#### Memory usage creeping up during long training
- **Symptom**: Memory usage increases over time
- **Cause**: Python garbage collection issues with MPS
- **Solution**: Force garbage collection in training loop

```python
# Add to training loop
import gc
if step % 100 == 0:
    gc.collect()
    torch.mps.empty_cache()
```

### 3. Data Loading Issues

#### "Config name is missing" for C4 dataset
```
ValueError: Config name is missing. Please pick one among the available configs
```

**Solution:** Specify config in YAML:
```yaml
data:
  dataset: allenai/c4
  config: en  # Add this line
  split: train
```

#### Slow data loading or tokenization
- **Check**: Network speed, disk I/O
- **Solutions**:
  - Use SSD storage for data
  - Pre-tokenize data with `stream_for_token_budget.py`
  - Reduce `num_workers` in config
  - Use `streaming: true` for large datasets

### 4. Training Stability Issues

#### Loss not decreasing (stuck at high values)
```
Loss: 708.4653, not decreasing after 100+ steps
```

**Check these settings:**
```yaml
# Learning rate too low
lr: 3e-4  # Try 1e-3 for small models

# Inadequate warmup
warmup_steps: 200  # Increase to 1000 for stability

# Batch size too small
grad_accum_steps: 4  # Increase effective batch size

# Optimizer issues
momentum: 0.95  # Muon momentum (keep stable)
```

#### NaN losses appearing
```
Loss: nan, PPL: nan
```

**Immediate actions:**
1. Reduce learning rate by 10x
2. Enable gradient clipping: `max_grad_norm: 1.0`
3. Check for invalid data in dataset
4. Restart training with better initialization

#### Training crashes with "illegal memory access"
- **Cause**: MPS backend bug or memory corruption
- **Solutions**:
  - Reduce batch size to 1
  - Disable compilation: `compile: false`
  - Restart Python session
  - Try different PyTorch version

### 5. Configuration Issues

#### YAML parsing errors
```
yaml.YAMLError: mapping values are not allowed here
```

**Common mistakes:**
- Wrong indentation (use spaces, not tabs)
- Missing colons after keys
- Scientific notation issues (use quotes): `lr: "3e-4"`

#### Model size mismatches
```
RuntimeError: size mismatch between tensors
```

**Check:**
- `vocab_size` matches tokenizer vocabulary
- Model dimensions are consistent
- RoPE `max_seq_len` ≥ training `seq_len`

### 6. Performance Issues

#### Training too slow (< 500 tokens/sec)
```bash
# Check current performance
python -c "
import torch
import time
x = torch.randn(1, 1024, 768).to('mps')
start = time.time()
for _ in range(100):
    y = x @ x.transpose(-1, -2)
    y.sum().backward()
torch.mps.synchronize()
print(f'MPS performance: {(time.time() - start) * 10:.2f}ms per step')
"
```

**Optimization steps:**
1. Use `torch.compile` if stable
2. Optimize data loading (reduce `num_workers`)
3. Use larger batches where possible
4. Check for CPU bottlenecks

### 7. Monitoring and Logging Issues

#### TensorBoard not showing metrics
```bash
# Check TensorBoard installation
pip install tensorboard

# Launch TensorBoard
tensorboard --logdir checkpoints/pretrain/logs/tensorboard

# Check for events files
ls -la checkpoints/pretrain/logs/tensorboard/
```

#### Missing training logs
- Check file permissions
- Ensure `log_path` directory exists
- Verify disk space available

### 8. Tokenizer Issues

#### "Tokenizer model file not found"
```
FileNotFoundError: data/tokenizer.model
```

**Solutions:**
```bash
# Train new tokenizer
python scripts/train_tokenizer.py \
  --input data/ultrafineweb_sample.txt \
  --vocab-size 32000 \
  --model-prefix data/tokenizer

# Or download existing tokenizer
# (if you have a backup)
```

#### Poor tokenization quality (high compression ratio)
- Retrain on larger corpus (30M+ tokens)
- Use domain-specific data
- Increase vocabulary size

### 9. Checkpoint Issues

#### Cannot resume from checkpoint
```
KeyError: 'optimizer_state_dict'
```

**Check checkpoint compatibility:**
- Ensure same model architecture
- Same optimizer configuration
- Same PyTorch version

#### Checkpoint rotation not working
- Check `ckpt_keep` setting
- Ensure write permissions
- Monitor disk space

### 10. Environment Issues

#### Virtual environment problems
```bash
# Recreate virtual environment
deactivate
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

#### Dependency conflicts
```bash
# Check conflicting packages
pip check

# Reinstall in clean environment
pip freeze > requirements.txt
pip uninstall -r requirements.txt -y
pip install -e .
```

## Getting Help

### Debug Information to Include
When reporting issues, please include:

```bash
# System information
python -c "
import torch, sys
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
print('MPS built:', torch.backends.mps.is_built())
"

# Training configuration
cat configs/pretrain.yaml

# Recent training logs
tail -50 checkpoints/pretrain/logs/pretrain_768d_12l_metrics.jsonl

# Final training summary
cat checkpoints/pretrain/logs/final_training_summary.json
```

### Performance Benchmarks

Expected performance on M3 Pro 18GB:
- **Throughput**: 800-1200 tokens/sec
- **Memory usage**: 2-3GB during training
- **Loss reduction**: 10-20% per 1000 steps (initial training)
- **Stability**: No crashes during 24+ hour runs

### Support Resources
- GitHub Issues: Report bugs with full debug info
- Documentation: Check `docs/` for detailed guides
- Training logs: Include performance metrics when reporting issues
