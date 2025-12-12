#!/usr/bin/env python
"""
Performance benchmarking script for AnarchoBot.
Tests throughput, memory usage, and stability across different configurations.
"""
import argparse
import time
import torch
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anarchobot.config import ModelConfig
from anarchobot.model import TransformerLM
from anarchobot.memory_monitor import MemoryMonitor
from anarchobot.tokenizer import SentencePieceTokenizer
from anarchobot.utils import get_device


def benchmark_model_creation(model_sizes, seq_len=1024):
    """Benchmark model instantiation time and parameter count."""
    print("🧠 Model Creation Benchmark")
    print("-" * 50)

    device = get_device()
    results = []

    for size_name, (n_layers, d_model, n_heads) in model_sizes.items():
        try:
            start_time = time.time()
            config = ModelConfig(
                vocab_size=32000,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=4096
            )
            model = TransformerLM(config)
            model = model.to(device)
            creation_time = time.time() - start_time

            params = sum(p.numel() for p in model.parameters())
            params_m = params / 1e9

            results.append((size_name, params, params_m, creation_time))
            print("4s")

        except Exception as e:
            print("4s")

    return results


def benchmark_memory_usage(model, seq_len=1024, batch_size=1):
    """Benchmark memory usage during forward pass."""
    print(f"\n💾 Memory Usage Benchmark (seq_len={seq_len}, batch_size={batch_size})")
    print("-" * 50)

    device = get_device()
    monitor = MemoryMonitor(device)

    model.eval()
    torch.mps.empty_cache() if device.type == "mps" else torch.cuda.empty_cache()

    # Create input
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len)).to(device)

    # Measure memory before
    mem_before = monitor.get_memory_stats()

    try:
        with torch.no_grad():
            start_time = time.time()
            logits, loss = model(x, x)  # Use same tensor for input/target
            torch.mps.synchronize() if device.type == "mps" else None
            inference_time = time.time() - start_time

        # Measure memory after
        mem_after = monitor.get_memory_stats()

        mem_used = mem_after.gpu_memory_used - mem_before.gpu_memory_used
        tokens_per_sec = (batch_size * seq_len) / inference_time

        print("12.0f"
        print("8.0f"
        print("6.2f"
        if loss is not None:
            print("8.4f"
        return mem_used, tokens_per_sec

    except Exception as e:
        print(f"❌ Failed: {e}")
        return None, None


def benchmark_generation(model, tokenizer, prompts, max_new_tokens=50):
    """Benchmark text generation speed."""
    print("
🚀 Text Generation Benchmark"    print("-" * 50)

    device = next(model.parameters()).device
    model.eval()

    total_time = 0
    total_tokens = 0

    for prompt in prompts:
        try:
            tokens = torch.tensor([tokenizer.encode(prompt)]).to(device)

            start_time = time.time()
            with torch.no_grad():
                generated = model.generate(tokens, max_new_tokens=max_new_tokens, temperature=0.8)
            torch.mps.synchronize() if device.type == "mps" else None
            gen_time = time.time() - start_time

            generated_text = tokenizer.decode(generated[0].tolist())
            new_tokens = generated.size(1) - tokens.size(1)

            tokens_per_sec = new_tokens / gen_time
            total_time += gen_time
            total_tokens += new_tokens

            print("20s")
            print("20s")

        except Exception as e:
            print("20s")

    if total_time > 0:
        avg_tokens_per_sec = total_tokens / total_time
        print("20s")


def run_full_benchmark():
    """Run complete benchmark suite."""
    print("🚀 AnarchoBot Performance Benchmark")
    print("=" * 60)
    print(f"Hardware: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 60)

    # Model sizes to test
    model_sizes = {
        "tiny": (6, 256, 8),      # ~7M params
        "small": (8, 512, 8),     # ~30M params
        "medium": (12, 768, 12),  # ~137M params
    }

    # Create medium model for detailed testing
    config = ModelConfig(
        vocab_size=32000,
        n_layers=12,
        d_model=768,
        n_heads=12,
        max_seq_len=4096
    )
    model = TransformerLM(config).to(get_device())

    # Benchmarks
    creation_results = benchmark_model_creation(model_sizes)

    print("
📊 Benchmark Summary"    print("=" * 60)

    # Memory and throughput tests
    memory_usage, throughput = benchmark_memory_usage(model, seq_len=1024, batch_size=1)
    benchmark_memory_usage(model, seq_len=2048, batch_size=1)
    benchmark_memory_usage(model, seq_len=1024, batch_size=2)

    # Generation test (if tokenizer available)
    try:
        tokenizer = SentencePieceTokenizer("data/tokenizer.model")
        prompts = [
            "The future of AI is",
            "Apple Silicon Macs are",
            "Machine learning models"
        ]
        benchmark_generation(model, tokenizer, prompts)
    except:
        print("\n⚠️  Skipping generation benchmark (tokenizer not found)")

    print("
✅ Benchmark complete!"    print("For best performance:")
    print("- Keep micro_batch_size = 1")
    print("- Use gradient_checkpointing for seq_len > 1024")
    print("- Monitor memory usage with memory_monitor.py")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark AnarchoBot performance")
    parser.add_argument("--model-size", choices=["tiny", "small", "medium"],
                       default="medium", help="Model size to benchmark")
    parser.add_argument("--seq-len", type=int, default=1024,
                       help="Sequence length for testing")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for testing")
    parser.add_argument("--full", action="store_true",
                       help="Run full benchmark suite")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.full:
        run_full_benchmark()
    else:
        # Quick benchmark for specific config
        device = get_device()
        config = ModelConfig(
            vocab_size=32000,
            n_layers=12,
            d_model=768,
            n_heads=12,
            max_seq_len=4096
        )
        model = TransformerLM(config).to(device)

        print(f"Quick Benchmark: {args.model_size} model, seq_len={args.seq_len}, batch_size={args.batch_size}")
        benchmark_memory_usage(model, args.seq_len, args.batch_size)
