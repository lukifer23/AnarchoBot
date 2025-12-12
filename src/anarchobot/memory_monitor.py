"""
Memory monitoring utilities for AnarchoBot training on Apple Silicon.
"""
import torch
import psutil
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class MemoryStats:
    """Memory statistics for a training step."""
    step: int
    gpu_memory_used: float  # MB
    gpu_memory_peak: float  # MB
    gpu_memory_allocated: float  # MB
    gpu_memory_reserved: float  # MB
    system_memory_used: float  # MB
    system_memory_percent: float
    timestamp: float


class MemoryMonitor:
    """
    Monitor memory usage during training on Apple Silicon MPS.
    Tracks both GPU and system memory usage.
    """

    def __init__(self, device: torch.device, log_interval: int = 10):
        self.device = device
        self.log_interval = log_interval
        self.stats_history: List[MemoryStats] = []
        self.start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        self.end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

        # M3 Pro memory limits (in MB)
        self.gpu_memory_limit = 18 * 1024  # 18GB
        self.system_memory_limit = psutil.virtual_memory().total / (1024 * 1024)  # Total system RAM

        # Warning thresholds
        self.gpu_warning_threshold = 0.8  # 80% of GPU memory
        self.system_warning_threshold = 0.9  # 90% of system memory

    def get_memory_stats(self, step: int = 0) -> MemoryStats:
        """Get current memory statistics."""
        import time

        # GPU memory stats (MPS)
        if self.device.type == "mps":
            # MPS doesn't have direct memory queries like CUDA
            # Use torch.mps.current_allocated_memory() and torch.mps.driver_allocated_memory()
            try:
                gpu_allocated = torch.mps.current_allocated_memory() / (1024 * 1024)  # MB
                gpu_reserved = torch.mps.driver_allocated_memory() / (1024 * 1024)  # MB
                gpu_used = gpu_allocated  # Approximation
                gpu_peak = gpu_reserved  # Approximation
            except AttributeError:
                # Fallback if MPS memory functions aren't available
                gpu_allocated = gpu_reserved = gpu_used = gpu_peak = 0.0
        elif self.device.type == "cuda":
            gpu_used = torch.cuda.memory_used(self.device) / (1024 * 1024)
            gpu_allocated = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            gpu_reserved = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
            gpu_peak = 0.0  # Would need to track manually
        else:
            gpu_used = gpu_allocated = gpu_reserved = gpu_peak = 0.0

        # System memory stats
        system_stats = psutil.virtual_memory()
        system_used = system_stats.used / (1024 * 1024)
        system_percent = system_stats.percent

        return MemoryStats(
            step=step,
            gpu_memory_used=gpu_used,
            gpu_memory_peak=gpu_peak,
            gpu_memory_allocated=gpu_allocated,
            gpu_memory_reserved=gpu_reserved,
            system_memory_used=system_used,
            system_memory_percent=system_percent,
            timestamp=time.time()
        )

    def check_memory_limits(self, stats: MemoryStats) -> Tuple[bool, str]:
        """Check if memory usage is within safe limits."""
        warnings = []

        if stats.gpu_memory_used > self.gpu_memory_limit * self.gpu_warning_threshold:
            pct = (stats.gpu_memory_used / self.gpu_memory_limit) * 100
            warnings.append(f"GPU {stats.gpu_memory_used:.1f}MB ({pct:.1f}% of 18GB)")

        if stats.system_memory_percent > self.system_warning_threshold * 100:
            warnings.append(f"System {stats.system_memory_percent:.1f}%")

        if stats.gpu_memory_used > self.gpu_memory_limit * 0.95:
            return False, f"GPU {stats.gpu_memory_used:.1f}MB critical"

        if stats.system_memory_percent > 95:
            return False, f"System {stats.system_memory_percent:.1f}% critical"

        if warnings:
            return True, " | ".join(warnings)
        return True, ""

    def log_memory_stats(self, stats: MemoryStats, prefix: str = ""):
        """Log memory statistics."""
        safe, warnings = self.check_memory_limits(stats)

        status = "⚠️ " if warnings else "✅"

        log_msg = (
            f"{prefix}{status} Step {stats.step} | "
            f"GPU used {stats.gpu_memory_used:.1f}MB (alloc {stats.gpu_memory_allocated:.1f}MB, res {stats.gpu_memory_reserved:.1f}MB) | "
            f"System {stats.system_memory_used:.1f}MB ({stats.system_memory_percent:.1f}%)"
        )

        if warnings:
            log_msg += f" | Warnings: {warnings}"

        print(log_msg)

        if not safe:
            print("🚨 CRITICAL: Memory usage too high! Consider reducing batch size or enabling gradient checkpointing.")

        self.stats_history.append(stats)

    @contextmanager
    def monitor_step(self, step: int):
        """Context manager to monitor a training step."""
        if self.start_time:
            self.start_time.record()

        initial_stats = self.get_memory_stats(step)
        yield initial_stats

        if self.end_time:
            self.end_time.record()
            torch.cuda.synchronize()

        final_stats = self.get_memory_stats(step)

        if step % self.log_interval == 0:
            self.log_memory_stats(final_stats, "Memory: ")

    def get_memory_summary(self) -> Dict[str, float]:
        """Get summary statistics of memory usage."""
        if not self.stats_history:
            return {}

        gpu_used = [s.gpu_memory_used for s in self.stats_history]
        system_used = [s.system_memory_percent for s in self.stats_history]

        return {
            "gpu_memory_avg_mb": sum(gpu_used) / len(gpu_used),
            "gpu_memory_max_mb": max(gpu_used),
            "system_memory_avg_percent": sum(system_used) / len(system_used),
            "system_memory_max_percent": max(system_used),
            "total_measurements": len(self.stats_history)
        }

    def estimate_training_memory(self, model, batch_size: int, seq_len: int) -> Dict[str, float]:
        """Estimate memory requirements for training."""
        # Rough estimation based on model size and batch parameters

        # Model parameters
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # MB

        # Optimizer states (Adam has 2 states per parameter)
        optimizer_memory = param_memory * 2

        # Gradients
        gradient_memory = param_memory

        # Activations (rough estimate: 2x model size for forward + backward)
        activation_memory = param_memory * 2

        # Batch data (assuming int32 tokens)
        vocab_size = getattr(model.config if hasattr(model, 'config') else model, 'vocab_size', 32000)
        data_memory = batch_size * seq_len * 4 / (1024 * 1024)  # MB (4 bytes per token)

        total_estimated = param_memory + optimizer_memory + gradient_memory + activation_memory + data_memory

        return {
            "parameter_memory_mb": param_memory,
            "optimizer_memory_mb": optimizer_memory,
            "gradient_memory_mb": gradient_memory,
            "activation_memory_mb": activation_memory,
            "data_memory_mb": data_memory,
            "total_estimated_mb": total_estimated,
            "gpu_limit_mb": self.gpu_memory_limit,
            "usage_percent": (total_estimated / self.gpu_memory_limit) * 100
        }


def optimize_batch_size_for_memory(
    model,
    seq_len: int,
    target_memory_percent: float = 0.7,
    min_batch_size: int = 1,
    max_batch_size: int = 16
) -> int:
    """
    Find optimal batch size that stays within memory limits.

    Returns the largest batch size that keeps memory usage below target_memory_percent.
    """
    monitor = MemoryMonitor(torch.device("mps"))

    for batch_size in range(max_batch_size, min_batch_size - 1, -1):
        estimate = monitor.estimate_training_memory(model, batch_size, seq_len)
        if estimate["usage_percent"] <= target_memory_percent * 100:
            return batch_size

    return min_batch_size


# Utility functions for memory debugging
def log_tensor_memory_info(tensor: torch.Tensor, name: str = ""):
    """Log memory information for a tensor."""
    if tensor.is_cuda or tensor.device.type == "mps":
        memory_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        print(f"{name} tensor: {memory_mb:.2f} MB on {tensor.device}")
def garbage_collect():
    """Force garbage collection and MPS memory cleanup."""
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def profile_memory_usage(func, *args, **kwargs):
    """Profile memory usage of a function."""
    monitor = MemoryMonitor(torch.device("mps"))

    initial_stats = monitor.get_memory_stats()
    print("Initial memory state:")
    monitor.log_memory_stats(initial_stats)

    result = func(*args, **kwargs)

    final_stats = monitor.get_memory_stats()
    print("\nFinal memory state:")
    monitor.log_memory_stats(final_stats)

    return result
