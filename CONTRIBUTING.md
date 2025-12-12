# Contributing to AnarchoBot

Thank you for your interest in contributing to AnarchoBot! This document provides guidelines and information for contributors.

## Code of Conduct

This project follows a simple code of conduct: be respectful, constructive, and focused on technical excellence. No drama, no politics - just good code.

## Development Setup

### Prerequisites
- Apple Silicon Mac (M3/M4)
- Python 3.10+
- PyTorch with MPS support

### Installation
```bash
git clone https://github.com/lukifer23/AnarchoBot.git
cd AnarchoBot
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Verification
```bash
python scripts/setup_verification.py
```

## Development Workflow

### 1. Choose an Issue
- Check [GitHub Issues](https://github.com/lukifer23/AnarchoBot/issues) for open tasks
- Look for issues tagged `good first issue` or `help wanted`
- Comment on the issue to indicate you're working on it

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Make Changes
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 4. Test Your Changes
```bash
# Run setup verification
python scripts/setup_verification.py

# Run a minimal training test
PYTHONPATH=src python -m anarchobot.train --config configs/minimal_test.yaml

# Check memory usage
python -c "from anarchobot.memory_monitor import MemoryMonitor; print('Memory monitoring works')"
```

### 5. Commit and Push
```bash
git add .
git commit -m "Brief description of changes"
git push origin your-branch-name
```

### 6. Create Pull Request
- Go to [Pull Requests](https://github.com/lukifer23/AnarchoBot/pulls)
- Create a new PR with a clear title and description
- Reference any related issues
- Wait for review

## Code Style Guidelines

### Python Style
- Follow PEP 8
- Use type hints for function parameters and return values
- Use descriptive variable names
- Keep functions focused and under 50 lines when possible

### Documentation
- Add docstrings to all public functions
- Update README.md for user-facing changes
- Add comments for complex logic
- Keep documentation in the `docs/` directory

### Configuration
- Use YAML for configuration files
- Document all configuration parameters
- Provide sensible defaults
- Validate configuration at startup

## Architecture Guidelines

### Model Components
- Keep the model architecture modular
- Use clear separation between attention, MLP, and normalization
- Support different model sizes through configuration
- Maintain compatibility with standard transformer implementations

### Training Infrastructure
- Design for resumability from any checkpoint
- Include comprehensive error handling
- Provide detailed logging and monitoring
- Optimize for Apple Silicon MPS

### Data Pipeline
- Support streaming datasets for large-scale training
- Handle different data formats gracefully
- Provide data validation and quality checks
- Enable efficient tokenization and batching

## Testing

### Unit Tests
```bash
# Add tests in a tests/ directory
# Use pytest framework
python -m pytest tests/
```

### Integration Tests
```bash
# Test end-to-end training
PYTHONPATH=src python -m anarchobot.train --config tests/test_config.yaml
```

### Performance Tests
```bash
# Benchmark throughput
python scripts/benchmark.py --config configs/benchmark.yaml
```

## Areas for Contribution

### High Priority
- **Memory Optimization**: Improve memory efficiency for larger models
- **Multi-GPU Support**: Extend to M3 Ultra and Max systems
- **Model Architectures**: Add new transformer variants (Mamba, RWKV, etc.)
- **Evaluation Suite**: Add comprehensive model evaluation tools

### Medium Priority
- **Data Quality**: Improve data filtering and preprocessing
- **Checkpoint Management**: Better checkpoint compression and storage
- **Inference Optimization**: Faster generation for chat applications
- **Documentation**: More tutorials and examples

### Low Priority
- **Web Interface**: Simple training monitoring dashboard
- **Model Conversion**: Export to other formats (GGUF, etc.)
- **Benchmarking**: Compare against other implementations
- **CI/CD**: Automated testing and deployment

## Reporting Issues

### Bug Reports
When reporting bugs, please include:
- Full error traceback
- Configuration used
- Hardware specifications
- Steps to reproduce
- Expected vs actual behavior

### Feature Requests
For new features, please include:
- Use case and motivation
- Proposed implementation approach
- Potential impact on existing code
- Alternative solutions considered

### Performance Issues
For performance problems, include:
- Hardware specifications
- Configuration used
- Performance metrics (tokens/sec, memory usage)
- Expected vs actual performance

## Getting Help

- **GitHub Issues**: For bugs, features, and general discussion
- **Documentation**: Check `docs/` directory first
- **Code Review**: Ask questions in your PR
- **Architecture**: Refer to `docs/architecture.md`

## Recognition

Contributors will be recognized in:
- GitHub repository contributors list
- CHANGELOG.md for significant contributions
- Acknowledgments in documentation

## License

By contributing to AnarchoBot, you agree that your contributions will be licensed under the MIT License.
