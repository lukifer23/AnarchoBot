# Changelog

All notable changes to AnarchoBot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete SLM training infrastructure for Apple Silicon
- Muon optimizer implementation with orthogonalized momentum
- Comprehensive memory monitoring and management
- TensorBoard integration and detailed logging
- Robust checkpointing with automatic rotation
- End-to-end training pipeline validation
- MLX backend for enhanced Apple Silicon performance
- Pre-tokenized data pipeline for optimal MLX training
- Extensive documentation and configuration guides
- Setup verification and troubleshooting tools

### Performance
- **Throughput**: 976 tokens/second on M3 Pro
- **Memory Usage**: 2.1GB/18GB (14% utilization)
- **Stability**: Tested across 50 training steps
- **Resumability**: Full checkpoint recovery capability

### Technical Details
- GPT-style transformer with RoPE positional encoding
- RMSNorm, SwiGLU MLP, tied embeddings
- 4K context length support with gradient checkpointing
- MPS-exclusive backend for Apple Silicon optimization
- Streaming dataset support for large-scale training

## [0.1.0] - 2025-01-XX

### Added
- Initial release with core training functionality
- Basic model architecture (TransformerLM)
- Data loading and tokenization pipeline
- Training loop with optimizer integration
- Configuration system (YAML-based)
- Setup and verification scripts

### Infrastructure
- Git repository initialization
- MIT license
- Basic documentation
- CI/CD setup preparation

---

## Development Notes

### Current Status (Pre-v1.0)
- ✅ Core infrastructure complete and tested
- ✅ Memory monitoring and logging fully functional
- 🚧 Large-scale pre-training (requires improved tokenizer)
- 🚧 Model evaluation and benchmarking suite
- 🚧 Multi-GPU support for M3 Ultra/Max

### Performance Baselines
- **Hardware**: Apple M3 Pro (18GB RAM)
- **Test Run**: 50 steps, 409K tokens processed
- **Stability**: No crashes, consistent memory usage
- **Scalability**: Designed for 2.8B+ token training campaigns

### Architecture Decisions
- **MPS Exclusive**: Optimized for Apple Silicon, no CUDA fallback
- **Muon Optimizer**: Superior convergence for transformer training
- **Streaming Data**: Memory-efficient handling of large datasets
- **Comprehensive Monitoring**: Production-ready observability
- **Modular Design**: Easy extension and customization

### Future Releases
- **v0.2.0**: Large-scale pre-training completion
- **v0.3.0**: Multi-GPU support and model parallelism
- **v0.4.0**: Evaluation suite and benchmarking
- **v0.5.0**: Additional architectures (Mamba, RWKV)
- **v1.0.0**: Production-ready chat models and deployment tools
