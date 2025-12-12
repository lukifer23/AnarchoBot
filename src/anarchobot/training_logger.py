"""
Enhanced training logger for AnarchoBot with comprehensive metrics and monitoring.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import torch
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics for a step."""
    step: int
    loss: float
    learning_rate: float
    grad_norm: Optional[float] = None
    tokens_processed: int = 0
    throughput_tokens_per_sec: float = 0.0
    step_time_sec: float = 0.0
    epoch: int = 0
    timestamp: float = 0.0

    # Memory metrics
    gpu_memory_used_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    system_memory_percent: float = 0.0

    # Model metrics (optional)
    perplexity: Optional[float] = None
    param_norm: Optional[float] = None
    grad_norm_clipped: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrainingLogger:
    """
    Comprehensive training logger with multiple output formats.
    """

    def __init__(
        self,
        log_dir: Path,
        experiment_name: str = "anarchobot_training",
        log_to_tensorboard: bool = True,
        log_to_json: bool = True,
        console_log_interval: int = 25,
        checkpoint_log_interval: int = 500
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.console_log_interval = console_log_interval
        self.checkpoint_log_interval = checkpoint_log_interval

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = self.log_dir / "tensorboard"
        self.json_log_file = self.log_dir / f"{experiment_name}_metrics.jsonl"

        # Initialize loggers
        self.tensorboard_writer = SummaryWriter(self.tensorboard_dir) if log_to_tensorboard else None
        self.json_file = open(self.json_log_file, 'a') if log_to_json else None

        # Metrics history
        self.metrics_history: List[TrainingMetrics] = []
        self.start_time = time.time()

        # Performance tracking
        self.last_log_time = self.start_time
        self.total_tokens_processed = 0

    def log_step(self, metrics: TrainingMetrics):
        """Log metrics for a training step."""
        self.metrics_history.append(metrics)
        self.total_tokens_processed += metrics.tokens_processed

        # Console logging
        if metrics.step % self.console_log_interval == 0:
            self._log_to_console(metrics)

        # TensorBoard logging
        if self.tensorboard_writer:
            self._log_to_tensorboard(metrics)

        # JSON logging
        if self.json_file:
            self._log_to_json(metrics)

        # Periodic checkpoint logging
        if metrics.step % self.checkpoint_log_interval == 0:
            self._log_checkpoint_summary(metrics)

    def _log_to_console(self, metrics: TrainingMetrics):
        """Log metrics to console."""
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)

        # Format memory info
        if metrics.gpu_memory_used_mb > 0:
            mem_info = f"{metrics.gpu_memory_used_mb:.0f}MB GPU | {metrics.system_memory_percent:.1f}% system"
        else:
            mem_info = "N/A"

        # Format throughput
        if metrics.throughput_tokens_per_sec > 0:
            throughput_str = f"{metrics.throughput_tokens_per_sec:,.0f}"
        else:
            throughput_str = "N/A"

        # Format perplexity if available
        ppl_str = f"{metrics.perplexity:.2f}" if metrics.perplexity is not None else "N/A"

        print(
            f"Step {metrics.step:6d} | "
            f"Loss: {metrics.loss:.4f} | "
            f"PPL: {ppl_str} | "
            f"LR: {metrics.learning_rate:.6f} | "
            f"Time: {hours:02d}:{minutes:02d}:{seconds:02d} | "
            f"Tokens/sec: {throughput_str} | "
            f"Memory: {mem_info}"
        )

    def _log_to_tensorboard(self, metrics: TrainingMetrics):
        """Log metrics to TensorBoard."""
        writer = self.tensorboard_writer
        step = metrics.step

        # Core metrics
        writer.add_scalar('Loss/train', metrics.loss, step)
        writer.add_scalar('Learning_Rate', metrics.learning_rate, step)
        writer.add_scalar('Throughput/tokens_per_sec', metrics.throughput_tokens_per_sec, step)
        writer.add_scalar('Time/step_time', metrics.step_time_sec, step)

        # Memory metrics
        if metrics.gpu_memory_used_mb > 0:
            writer.add_scalar('Memory/gpu_used_mb', metrics.gpu_memory_used_mb, step)
            writer.add_scalar('Memory/gpu_peak_mb', metrics.gpu_memory_peak_mb, step)
        if metrics.system_memory_percent > 0:
            writer.add_scalar('Memory/system_percent', metrics.system_memory_percent, step)

        # Optional metrics
        if metrics.grad_norm is not None:
            writer.add_scalar('Gradients/norm', metrics.grad_norm, step)
        if metrics.perplexity is not None:
            writer.add_scalar('Metrics/perplexity', metrics.perplexity, step)
        if metrics.param_norm is not None:
            writer.add_scalar('Model/param_norm', metrics.param_norm, step)

    def _log_to_json(self, metrics: TrainingMetrics):
        """Log metrics to JSON Lines file."""
        if self.json_file:
            json.dump(metrics.to_dict(), self.json_file)
            self.json_file.write('\n')
            self.json_file.flush()

    def _log_checkpoint_summary(self, metrics: TrainingMetrics):
        """Log detailed summary at checkpoints."""
        recent_metrics = self.metrics_history[-self.checkpoint_log_interval:]
        if len(recent_metrics) < 10:
            return

        # Calculate averages
        avg_loss = sum(m.loss for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput_tokens_per_sec for m in recent_metrics) / len(recent_metrics)
        avg_step_time = sum(m.step_time_sec for m in recent_metrics) / len(recent_metrics)

        summary = {
            "checkpoint_step": metrics.step,
            "avg_loss_last_checkpoint": avg_loss,
            "avg_throughput_last_checkpoint": avg_throughput,
            "avg_step_time_last_checkpoint": avg_step_time,
            "total_tokens_processed": self.total_tokens_processed,
            "elapsed_time_sec": time.time() - self.start_time
        }

        # Write checkpoint summary
        checkpoint_file = self.log_dir / f"checkpoint_{metrics.step}_summary.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"📊 Checkpoint summary saved to {checkpoint_file}")

    def log_hyperparameters(self, config: Dict[str, Any]):
        """Log training hyperparameters."""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_hparams(config, {})

        # Also save to JSON
        hyperparams_file = self.log_dir / "hyperparameters.json"
        with open(hyperparams_file, 'w') as f:
            json.dump(config, f, indent=2)

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.metrics_history:
            return {}

        all_losses = [m.loss for m in self.metrics_history]
        all_throughput = [m.throughput_tokens_per_sec for m in self.metrics_history if m.throughput_tokens_per_sec > 0]

        summary = {
            "total_steps": len(self.metrics_history),
            "total_tokens_processed": self.total_tokens_processed,
            "total_training_time_sec": time.time() - self.start_time,
            "final_loss": all_losses[-1],
            "best_loss": min(all_losses),
            "avg_loss": sum(all_losses) / len(all_losses),
            "loss_std": torch.std(torch.tensor(all_losses)).item() if len(all_losses) > 1 else 0.0,
            "avg_throughput_tokens_per_sec": sum(all_throughput) / len(all_throughput) if all_throughput else 0.0,
            "peak_throughput_tokens_per_sec": max(all_throughput) if all_throughput else 0.0,
        }

        return summary

    def save_final_summary(self):
        """Save final training summary."""
        summary = self.get_training_summary()
        summary_file = self.log_dir / "final_training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print("📈 Final training summary saved!")
        print(f"   Total steps: {summary['total_steps']}")
        print(f"   Final loss: {summary['final_loss']:.4f}")
        print(f"   Best loss: {summary['best_loss']:.4f}")
        print(f"   Average throughput: {summary['avg_throughput_tokens_per_sec']:.0f} tokens/sec")

    def close(self):
        """Close all loggers."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        if self.json_file:
            self.json_file.close()


class TrainingProgressTracker:
    """
    Track training progress with ETA calculations and early stopping.
    """

    def __init__(self, total_steps: int, early_stopping_patience: Optional[int] = None):
        self.total_steps = total_steps
        self.early_stopping_patience = early_stopping_patience
        self.best_loss = float('inf')
        self.best_step = 0
        self.no_improvement_count = 0
        self.start_time = time.time()

    def should_stop_early(self, current_loss: float, current_step: int) -> bool:
        """Check if training should stop early."""
        if self.early_stopping_patience is None:
            return False

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_step = current_step
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        return self.no_improvement_count >= self.early_stopping_patience

    def get_eta(self, current_step: int) -> str:
        """Calculate estimated time to completion."""
        if current_step == 0:
            return "Unknown"

        elapsed = time.time() - self.start_time
        steps_per_sec = current_step / elapsed
        remaining_steps = self.total_steps - current_step
        eta_sec = remaining_steps / steps_per_sec

        hours, remainder = divmod(int(eta_sec), 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def get_progress_summary(self, current_step: int) -> Dict[str, Any]:
        """Get training progress summary."""
        progress_percent = (current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time

        return {
            "current_step": current_step,
            "total_steps": self.total_steps,
            "progress_percent": progress_percent,
            "elapsed_time_sec": elapsed,
            "eta": self.get_eta(current_step),
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "no_improvement_count": self.no_improvement_count
        }
