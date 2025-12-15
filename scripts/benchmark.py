#!/usr/bin/env python
"""
Performance benchmarking script for AnarchoBot (Apple Silicon).
Reports model creation time, forward memory usage, and generation speed.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anarchobot.config import ModelConfig  # noqa: E402
from anarchobot.memory_monitor import MemoryMonitor  # noqa: E402
from anarchobot.model import TransformerLM  # noqa: E402
from anarchobot.tokenizer import SentencePieceTokenizer  # noqa: E402
from anarchobot.utils import get_device  # noqa: E402


def benchmark_model_creation(model_sizes):
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
                max_seq_len=4096,
            )
            model = TransformerLM(config).to(device)
            creation_time = time.time() - start_time
            params = sum(p.numel() for p in model.parameters())
            results.append((size_name, params, creation_time))
            print(f"{size_name:>6} | {params/1e6:8.1f}M params | create {creation_time*1000:6.1f} ms")
        except Exception as e:
            print(f"{size_name:>6} | failed: {e}")

    return results


def benchmark_memory_usage(model, seq_len=1024, batch_size=1):
    """Benchmark memory usage during forward pass."""
    print(f"\n💾 Memory Usage Benchmark (seq_len={seq_len}, batch_size={batch_size})")
    print("-" * 50)

    device = get_device()
    monitor = MemoryMonitor(device)

    model.eval()
    torch.mps.empty_cache() if device.type == "mps" else torch.cuda.empty_cache()

    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len)).to(device)
    mem_before = monitor.get_memory_stats()

    try:
        with torch.no_grad():
            start_time = time.time()
            logits, loss = model(x, x)
            if device.type == "mps":
                torch.mps.synchronize()
            inference_time = time.time() - start_time

        mem_after = monitor.get_memory_stats()
        mem_used = mem_after.gpu_memory_used - mem_before.gpu_memory_used
        tokens_per_sec = (batch_size * seq_len) / max(inference_time, 1e-9)

        print(f"Used {mem_used:.1f} MB | {tokens_per_sec:,.0f} tok/s | time {inference_time*1000:.1f} ms")
        return mem_used, tokens_per_sec
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None, None


def benchmark_generation(model, tokenizer, prompts, max_new_tokens=50):
    """Benchmark text generation speed."""
    print("\n🚀 Text Generation Benchmark")
    print("-" * 50)

    device = next(model.parameters()).device
    model.eval()

    total_time = 0.0
    total_tokens = 0

    for prompt in prompts:
        try:
            tokens = torch.tensor([tokenizer.encode(prompt)]).to(device)

            start_time = time.time()
            with torch.no_grad():
                generated = model.generate(tokens, max_new_tokens=max_new_tokens, temperature=0.8)
            if device.type == "mps":
                torch.mps.synchronize()
            gen_time = time.time() - start_time

            new_tokens = generated.size(1) - tokens.size(1)
            tokens_per_sec = new_tokens / max(gen_time, 1e-9)
            total_time += gen_time
            total_tokens += new_tokens

            generated_text = tokenizer.decode(generated[0].tolist())
            print(f"Prompt: {prompt[:40]}... | {tokens_per_sec:,.0f} tok/s | {gen_time*1000:.1f} ms")
            print(f"Generated: {generated_text[:120].replace(chr(10), ' ')}")
        except Exception as e:
            print(f"Failed on prompt '{prompt[:30]}...': {e}")

    if total_time > 0:
        avg_tokens_per_sec = total_tokens / total_time
        print(f"Avg generation throughput: {avg_tokens_per_sec:,.0f} tok/s")


def run_full_benchmark():
    """Run complete benchmark suite."""
    print("🚀 AnarchoBot Performance Benchmark")
    print("=" * 60)
    print(f"Hardware: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 60)

    model_sizes = {
        "tiny": (6, 256, 8),
        "small": (8, 512, 8),
        "medium": (12, 768, 12),
    }

    config = ModelConfig(
        vocab_size=32000,
        n_layers=12,
        d_model=768,
        n_heads=12,
        max_seq_len=4096,
    )
    model = TransformerLM(config).to(get_device())

    benchmark_model_creation(model_sizes)

    benchmark_memory_usage(model, seq_len=1024, batch_size=1)
    benchmark_memory_usage(model, seq_len=2048, batch_size=1)
    benchmark_memory_usage(model, seq_len=1024, batch_size=2)

    try:
        tokenizer = SentencePieceTokenizer("data/tokenizer.model")
        prompts = [
            "The future of AI is",
            "Apple Silicon Macs are",
            "Machine learning models",
        ]
        benchmark_generation(model, tokenizer, prompts)
    except Exception as e:
        print(f"\n⚠️  Skipping generation benchmark (tokenizer not found or failed: {e})")

    print("\n✅ Benchmark complete!")
    print("For best performance:")
    print("- Keep micro_batch_size = 1")
    print("- Use gradient_checkpointing for seq_len > 1024")
    print("- Monitor memory usage with memory_monitor.py")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark AnarchoBot performance")
    parser.add_argument("--model-size", choices=["tiny", "small", "medium"], default="medium", help="Model size to benchmark")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length for testing")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.full:
        run_full_benchmark()
    else:
        device = get_device()
        if args.model_size == "tiny":
            n_layers, d_model, n_heads = (6, 256, 8)
        elif args.model_size == "small":
            n_layers, d_model, n_heads = (8, 512, 8)
        else:
            n_layers, d_model, n_heads = (12, 768, 12)

        config = ModelConfig(
            vocab_size=32000,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=4096,
        )
        model = TransformerLM(config).to(device)

        print(f"Quick Benchmark: {args.model_size} model, seq_len={args.seq_len}, batch_size={args.batch_size}")
        benchmark_memory_usage(model, args.seq_len, args.batch_size)
