## AnarchoBot (Apple Silicon, Muon)

Minimal SLM training stack for Mac (M3 Pro, 18GB) using PyTorch on MPS and the Muon optimizer. No guardrails, designed for training from scratch with 4K context support, and pipelines for pretrain → SFT → preference tuning. Also supports MLX backend for optimized Apple Silicon performance.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Data      │ -> │   Tokenization  │ -> │   Pre-training  │ -> │   Fine-tuning   │
│  (Ultra-FineWeb)│    │  (SentencePiece)│    │   (Next-token)  │    │  (Supervised)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         v                       v                       v                       v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Tokenizer     │    │   137M Model   │    │   Checkpoints   │    │   Chat Model    │
│   (32K vocab)   │    │   (GPT-style)   │    │   (Rotating)    │    │   (Inference)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

**🚀 Current Status**: Infrastructure complete, tested end-to-end. Ready for large-scale training.

### Stack
- Model: GPT-style Transformer, RoPE positions, RMSNorm, SwiGLU MLP, tied embeddings, 4K max context.
- Optimizer: Muon (orthogonalized momentum) for matrix weights + AdamW for embeddings/bias; cosine LR + warmup.
- Backend: PyTorch on MPS or MLX (Apple Silicon optimized). No CUDA/CPU fallback; run on a Mac with MPS-enabled PyTorch or MLX installed. Gradient checkpointing to save memory.
- Data: HF `datasets` streaming; sentencepiece tokenizer trained locally. Pre-tokenized shards supported for MLX.
- RL: DPO-style preference tuning (beta-adjustable) on a reference-frozen copy.
- Monitoring: Comprehensive training metrics, memory monitoring, TensorBoard logging, progress tracking.

### Current Status
- ✅ Complete training infrastructure with memory monitoring and logging
- ✅ Successfully tested end-to-end training pipeline (50 steps completed)
- ✅ Memory usage: ~2.1GB/18GB on M3 Pro (excellent efficiency)
- ✅ Throughput: ~976 tokens/sec during training (PyTorch MPS)
- ✅ MLX backend implementation with pre-tokenized data support
- 🚧 Full pre-training (2.8B tokens) in progress
- 🚧 Model has not been fully trained yet (infrastructure ready, training ongoing)

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Tokenizer (SentencePiece, BPE)
1) Materialize a corpus sample (Ultra-FineWeb recommended for quality):
```bash
python scripts/stream_text_dataset.py --dataset EliMC/Ultra-FineWeb --split en --text-field content --samples 200000 --score-field score --score-min 0.8 --output data/ultrafineweb_sample.txt
```
2) Train tokenizer on larger sample (30M+ tokens recommended):
```bash
python scripts/train_tokenizer.py --input data/ultrafineweb_full/shard_00000.txt --vocab-size 32000 --model-prefix data/tokenizer
```
The `configs/*.yaml` files point at `data/tokenizer.model` by default. Keep `model.vocab_size` in sync with the trained tokenizer size.

### Pre-train (next-token LM)
Config: `configs/pretrain_unified.yaml` (canonical 12x768, 4K context, memmap shards). Use `configs/pretrain_mlx_smoke.yaml` for a fast MLX smoke profile.

**PyTorch MPS Backend:**
```bash
PYTHONPATH=src python -m anarchobot.train --config configs/pretrain_unified.yaml
```

**MLX Backend (requires pre-tokenized shards):**
```bash
PYTHONPATH=src python scripts/train_mlx.py --config configs/pretrain_unified.yaml --shard-dir data/pretrain_shards --format npy
```

Notes:
- PyTorch MPS: Runs on MPS if available; uses fp16 autocast.
- MLX: Optimized for Apple Silicon with better memory efficiency; gradient accumulation now matches PyTorch behavior.
- Muon LR defaults to `lr` and Adam LR to `1.5 * lr`. Adjust in config.
- Checkpoints land in `checkpoints/pretrain_unified/` (PyTorch) or `checkpoints/pretrain_mlx/` (MLX).
- Memory monitoring active - training uses ~2.1GB/18GB on M3 Pro.
- Comprehensive logging to TensorBoard and JSON files.
- Tested successfully on small scale (50 steps); full training (2.8B tokens) requires improved tokenizer first.

### Supervised Fine-tune (chatty data)
Config: `configs/sft.yaml` (Ultrachat 200k, `messages` field handled automatically).
```bash
python -m anarchobot.finetune --config configs/sft.yaml --base-checkpoint checkpoints/pretrain/step_2000.pt
```
Outputs go to `checkpoints/sft/`.

### Preference Tuning (DPO)
Config: `configs/rlhf.yaml` (UltraFeedback binarized; expects `prompt/chosen/rejected` fields). Run:
```bash
python -m anarchobot.rlhf --config configs/rlhf.yaml \
  --sft-checkpoint checkpoints/sft/sft_last.pt --beta 0.1
```
Produces DPO-tuned checkpoints in `checkpoints/rlhf/`.

### Chat
```bash
python scripts/chat.py --config configs/sft.yaml --checkpoint checkpoints/sft/sft_last.pt
```
Multi-turn prompt history is kept in plain text; there are no safety filters.

### Muon details
- Update: orthogonalizes momentum with a Newton–Schulz iteration (`optim.py`). Matrix-like params (ndim ≥ 2, excluding embeds) use Muon; everything else uses AdamW.
- Works on a single device (no distributed ops). Weight decay is decoupled.
- Use `lr`/`momentum` to tune Muon; keep `momentum≈0.95` as a stable default.

### Data knobs
- Pre-train: C4 stream works; for lean runs you can swap in `NeelNanda/pile-uncopyrighted` or `HuggingFaceFW/fineweb` subsets by editing `configs/pretrain.yaml`.
- SFT: `HuggingFaceH4/ultrachat_200k` is default; OpenAssistant (`OpenAssistant/oasst1`) also works because `messages` lists are flattened.
- DPO: `HuggingFaceH4/ultrafeedback_binarized` is configured; any dataset with `prompt/chosen/rejected` fields will flow.
- Alternate pretrain corpus: `EliMC/Ultra-FineWeb` (`content` string field, `en` split). Swap into `data.dataset` and set `data.text_field: content` to leverage its filtered web crawl.

### Unified data pipeline
- Plan token budget + shards from the config:  
  `python scripts/pretrain_data_pipeline.py plan --config configs/pretrain_unified.yaml --tokens-per-shard 50000000`
- Stream Ultra-FineWeb text until target tokens:  
  `python scripts/pretrain_data_pipeline.py stream --config configs/pretrain_unified.yaml --tokenizer data/tokenizer.model --output-dir data/raw_ultrafineweb --tokens-per-shard 50000000 --score-field score --score-min 0.8`
- Pretokenize shards for both MPS/MLX (memmap npy pairs):  
  `python scripts/pretrain_data_pipeline.py tokenize --input-dir data/raw_ultrafineweb --output-dir data/pretrain_shards --tokenizer data/tokenizer.model --seq-len 2048 --format npy --cleanup-text`
- Train tokenizer (32k BPE):  
  `python scripts/train_tokenizer.py --input data/raw_ultrafineweb/shard_00000.txt --vocab-size 32000 --model-prefix data/tokenizer_v2`
- Pretrain (PyTorch MPS):
  `PYTHONPATH=src python -m anarchobot.train --config configs/pretrain_unified.yaml`
- Pretrain (MLX):
  `PYTHONPATH=src python scripts/train_mlx.py --config configs/pretrain_unified.yaml --shard-dir data/pretrain_shards --format npy`
- SFT (Ultrachat):
  `PYTHONPATH=src python -m anarchobot.finetune --config configs/sft.yaml --base-checkpoint checkpoints/pretrain_unified/step_2000.pt`
- DPO (UltraFeedback):
  `PYTHONPATH=src python -m anarchobot.rlhf --config configs/rlhf.yaml --sft-checkpoint checkpoints/sft/sft_last.pt --beta 0.1`
- Model size + token target:
  `python scripts/model_stats.py --config configs/pretrain_unified.yaml`
- Performance benchmark:
  `python scripts/benchmark.py --full`

### Performance tips (M3 Pro 18GB)
- Keep `micro_batch_size=1` and rely on `grad_accum_steps` to hit target tokens/step.
- Enable gradient checkpointing (already on in configs) to fit 2–4K sequences.
- Use `data.seq_len=1024–2048` for most steps, then a brief 4K curriculum near the end.
- MPS has no bf16; leave `precision=float16` for stability. CUDA users can flip to bf16.

### Files of interest
- `src/anarchobot/model.py` – Transformer with RoPE/RMSNorm/SwiGLU.
- `src/anarchobot/optim.py` – Muon + AdamW hybrid optimizer.
- `src/anarchobot/train.py` – pretraining loop (streaming dataloader, cosine LR).
- `src/anarchobot/finetune.py` – SFT loop.
- `src/anarchobot/rlhf.py` – DPO preference tuner.
- `scripts/prepare_corpus.py` / `scripts/train_tokenizer.py` / `scripts/chat.py`.

This repo provides a minimal but robust training stack for Apple Silicon: comprehensive monitoring, memory management, and logging infrastructure for training small language models from scratch. No guardrails or safety filters included - use responsibly.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Current Status](#current-status)
- [Setup](#setup)
- [Training Pipeline](#training-pipeline)
- [Documentation](#documentation)
- [Key Features](#key-features)
- [Performance](#performance)

## Documentation

- **[Quick Start Guide](docs/quickstart.md)** - Get up and running in 15 minutes
- **[Architecture Overview](docs/architecture.md)** - Technical deep dive
- **[Configuration Guide](docs/configuration.md)** - Detailed parameter tuning
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[Examples](examples/README.md)** - Sample configs and usage patterns
- **[Contributing](CONTRIBUTING.md)** - Development guidelines
- **[Changelog](CHANGELOG.md)** - Version history

## Key Features

- **🧠 Muon Optimizer**: Orthogonalized momentum for stable, fast convergence
- **🍎 Apple Silicon Optimized**: MPS-exclusive with memory-efficient training
- **📊 Comprehensive Monitoring**: Real-time metrics, TensorBoard integration
- **🔄 Robust Resumption**: True checkpointing with automatic recovery
- **🎯 Production Ready**: Error handling, logging, and validation
- **📈 Scalable**: From tiny 7M models to large 500M+ parameter models

## Performance

| Hardware | Backend | Throughput | Memory | Context |
|----------|---------|------------|--------|---------|
| M3 Pro | PyTorch MPS | ~976 tokens/sec | 2.1GB | 4K |
| M3 Pro | MLX | ~2000+ tokens/sec | <2GB | 4K |
| Tested on | Both | 50 steps | 409K tokens | Stable |

*Full pre-training (2.8B tokens) pending improved tokenizer training. MLX provides better performance but requires pre-tokenized data.*

## Project Status

### ✅ Completed
- **Core Infrastructure**: Complete training stack with monitoring
- **Apple Silicon Optimization**: MPS-exclusive with memory management
- **Documentation**: Comprehensive guides and examples
- **Testing**: End-to-end pipeline validation
- **Open Source**: MIT licensed, contribution-ready

### 🚧 Next Steps
- **Large-Scale Training**: 2.8B token pre-training campaign
- **Improved Tokenizer**: 30M+ token training sample
- **Model Scaling**: Support for larger architectures
- **Multi-GPU**: M3 Ultra/Max support

### 🎯 Roadmap
- **v0.2.0**: Production chat models
- **v0.3.0**: Multi-GPU training
- **v0.4.0**: Model evaluation suite
- **v1.0.0**: Complete training platform

---

*Built with ❤️ on Apple Silicon. No guardrails, just good code.*
