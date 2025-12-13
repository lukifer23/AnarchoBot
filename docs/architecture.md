# AnarchoBot Architecture

## Overview

AnarchoBot is a complete small language model training stack optimized for Apple Silicon, featuring a GPT-style transformer architecture with the Muon optimizer and comprehensive monitoring infrastructure.

## Training Pipeline

```
Raw Data → Tokenizer → Pre-training → Supervised Fine-tuning → Preference Tuning → Inference
```

### 1. Data Processing
- **Input**: Streaming datasets from HuggingFace (C4, Ultra-FineWeb, etc.)
- **Tokenization**: SentencePiece BPE tokenizer trained on domain-specific corpus
- **Batching**: Dynamic batching with sequence packing for efficient GPU utilization

### 2. Model Architecture

```
┌─────────────────┐
│   Input Tokens  │
└─────────────────┘
         │
    ┌────────────┐
    │ Embeddings │ ← Tied with output
    └────────────┘
         │
    ┌────────────┐
    │ Transformer│ ← 12 layers
    │   Blocks   │
    └────────────┘
         │
    ┌────────────┐
    │  Language  │
    │   Head     │
    └────────────┘
         │
    ┌────────────┐
    │ Next Token │
    │ Prediction │
    └────────────┘
```

#### Transformer Block Details

```
Input → LayerNorm → Attention → Residual → LayerNorm → MLP → Residual → Output

Attention: Multi-Head Self-Attention with RoPE positional encoding
MLP: SwiGLU activation with expansion factor 4x
Norm: RMSNorm (ε=1e-5)
```

### 3. Optimizer Stack

```
Parameters → Routing → Muon (matrices) + AdamW (embeddings/bias)
                   │
            ┌──────┴──────┐
            │             │
        Muon Optimizer    AdamW Optimizer
        (orthogonalized   (standard)
         momentum)        │
                          │
                   ┌──────┴──────┐
                   │             │
               Cosine LR     Weight Decay
               Schedule      (decoupled)
```

#### Parameter Partitioning
- **Muon**: Parameters with `ndim ≥ 2` (weights, excluding embeddings)
- **AdamW**: 1D parameters (embeddings, biases, layer norms)

### 4. Memory Management

```
GPU Memory (18GB M3 Pro)
├── Model Parameters: ~500MB
├── Optimizer States: ~1GB
├── Activations: ~500MB
├── Gradients: ~500MB
└── Batch Data: ~100MB
    Total: ~2.6GB (14% utilization)
```

#### Gradient Checkpointing
- Reduces memory by 60-70% during training
- Trades computation for memory
- Essential for 4K context lengths

### 4. Backend Options

**PyTorch MPS Backend:**
- Metal Performance Shaders integration
- Automatic mixed precision (FP16)
- Gradient checkpointing for memory efficiency
- Comprehensive debugging and profiling

**MLX Backend:**
- Native Apple Silicon optimization
- Improved memory efficiency
- Higher throughput potential
- Pre-tokenized data format for optimal performance

### 5. Monitoring Infrastructure

```
Training Loop
    ├── Memory Monitor
    │   ├── GPU usage tracking
    │   ├── System memory monitoring
    │   └── Automatic batch size optimization
    │
    ├── Training Logger
    │   ├── TensorBoard integration
    │   ├── JSON metrics logging
    │   └── Progress tracking with ETA
    │
    └── Progress Tracker
        ├── Early stopping support
        └── Convergence monitoring
```

## Key Innovations

### Muon Optimizer
- Orthogonalizes momentum using Newton-Schulz iteration
- Superior convergence compared to AdamW for large parameter spaces
- Maintains stability during long training runs

### Apple Silicon Optimization
- MPS-exclusive design (no CUDA fallback)
- Memory-efficient training with gradient checkpointing
- Optimized for M3/M4 chip architectures

### Robust Training Infrastructure
- Comprehensive error handling and recovery
- Automatic checkpoint management with rotation
- Real-time performance monitoring
- Support for interrupted training resumption

## Performance Characteristics

- **Throughput**: ~976 tokens/second on M3 Pro
- **Memory Efficiency**: 14% GPU utilization for 137M parameter model
- **Context Length**: 4K tokens with gradient checkpointing
- **Training Stability**: Proven across 50+ training steps
- **Scalability**: Designed for 2.8B+ token training campaigns
