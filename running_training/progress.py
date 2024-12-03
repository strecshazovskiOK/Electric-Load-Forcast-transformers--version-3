import os
import platform
import time
from datetime import datetime, timedelta
import torch
import numpy as np
from utils.logging.logger import Logger
from evaluation.evaluator import Evaluator
from evaluation.metrics import MetricConfig
from typing import Optional

class Colors:
    """ANSI color codes with Windows support check"""
    def __init__(self):
        # Force enable colors on Windows
        if platform.system() == 'Windows':
            os.system('color')
        self.use_colors = True
        
        # Color codes remain the same
        self.HEADER = '\033[95m'
        self.BLUE = '\033[94m'
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.RED = '\033[91m'
        self.ENDC = '\033[0m'
        self.BOLD = '\033[1m'

class TrainingProgress:
    """Manages training progress tracking and logging."""
    
    def __init__(self, total_epochs: int, logger: Logger):
        self.total_epochs = total_epochs
        self.logger = logger
        self.start_time = time.time()
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        self.last_log_time = self.start_time
        self.log_interval = 5  # seconds
        
        # Initialize evaluator for metric calculations
        self.evaluator = Evaluator(
            metric_config=MetricConfig(
                resolution_minutes=60,
                rmse_threshold=0.05,
                mae_threshold=0.05,
                mape_threshold=0.05
            )
        )
        
        self.colors = Colors()
        self.separator = f"{self.colors.BLUE}{'═' * 100}{self.colors.ENDC}" if self.colors.use_colors else '=' * 100
        self.batch_separator = f"{self.colors.YELLOW}{'─' * 100}{self.colors.ENDC}" if self.colors.use_colors else '-' * 100
        self.current_epoch = 0
        self.total_batches = 0
        self.best_metrics = {
            'train_loss': float('inf'),
            'val_loss': float('inf'),
            'rmse': float('inf'),
            'mae': float('inf')
        }

    def _create_progress_bar(self, progress_pct: float) -> str:
        """Create a progress bar with simplified characters for better compatibility"""
        bar_length = 30
        filled_length = int(progress_pct / 100 * bar_length)
        
        if self.colors.use_colors:
            return (
                f"{self.colors.GREEN}{'#' * filled_length}"
                f"{self.colors.YELLOW}{'-' * (bar_length - filled_length)}{self.colors.ENDC}"
            )
        else:
            return '#' * filled_length + '>' + '-' * (bar_length - filled_length - 1)

    def log_batch(self, batch: int, total_batches: int, loss: float) -> None:
        """Log batch progress with improved formatting."""
        current_time = time.time()
        if self.total_batches == 0:
            self.total_batches = total_batches

        should_log = (current_time - self.last_log_time >= self.log_interval) or (batch == total_batches - 1)
        
        if should_log:
            progress_pct = (batch + 1) / total_batches * 100
            elapsed = current_time - self.start_time
            batches_per_sec = (batch + 1) / elapsed if elapsed > 0 else 0
            eta = timedelta(seconds=int((total_batches - batch - 1) / batches_per_sec)) if batches_per_sec > 0 else "Unknown"

            progress_bar = self._create_progress_bar(progress_pct)
            c = self.colors  # shorthand

            status = (
                f"\nEpoch Progress:\n"
                f"└─ Epoch: {c.GREEN}{self.current_epoch + 1}/{self.total_epochs}{c.ENDC}\n"
                f"└─ Batch: {c.BLUE}{batch + 1}/{total_batches}{c.ENDC}\n"
                f"└─ Progress: [{progress_bar}] {c.YELLOW}{progress_pct:.1f}%{c.ENDC}\n"
                f"└─ Loss: {c.RED}{loss:.6f}{c.ENDC}\n"
                f"└─ Speed: {c.BLUE}{batches_per_sec:.2f}{c.ENDC} batches/s\n"
                f"└─ Time: {c.YELLOW}{str(timedelta(seconds=int(elapsed)))}{c.ENDC} / "
                f"ETA: {c.GREEN}{str(eta)}{c.ENDC}\n"
            )
            
            self.logger.info(status)
            self.last_log_time = current_time

    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None, 
                 y_pred: Optional[torch.Tensor] = None, y_true: Optional[torch.Tensor] = None) -> None:
        """Log epoch summary with enhanced metrics."""
        self.current_epoch = epoch
        time_elapsed = time.time() - self.start_time
        remaining_epochs = self.total_epochs - (epoch + 1)
        time_per_epoch = time_elapsed / (epoch + 1) if epoch > 0 else 0
        eta = timedelta(seconds=int(time_per_epoch * remaining_epochs))

        # Update best metrics
        self.best_metrics['train_loss'] = min(self.best_metrics['train_loss'], train_loss)
        if val_loss is not None:
            self.best_metrics['val_loss'] = min(self.best_metrics['val_loss'], val_loss)

        # Calculate detailed metrics
        metrics = {}
        if y_pred is not None and y_true is not None:
            try:
                dummy_timestamps = np.arange(len(y_pred))
                eval_result = self.evaluator.evaluate(y_pred, y_true, dummy_timestamps)
                metrics = eval_result.total_metrics
                # Update best metrics
                for key in ['rmse', 'mae']:
                    if key in metrics:
                        self.best_metrics[key] = min(self.best_metrics[key], metrics[key])
            except Exception as e:
                self.logger.warning(f"Failed to calculate metrics: {str(e)}")

        # Create formatted summary
        header = (
            f"\n{self.separator}\n"
            f"{self.colors.BOLD}{self.colors.HEADER}║ Epoch {epoch + 1}/{self.total_epochs} Summary ║{self.colors.ENDC}\n"
            f"{self.separator}\n"
        )

        losses = (
            f"{self.colors.BOLD}Training Metrics:{self.colors.ENDC}\n"
            f"├─ Current Loss: {self.colors.RED}{train_loss:.6f}{self.colors.ENDC}\n"
            f"├─ Best Loss:    {self.colors.GREEN}{self.best_metrics['train_loss']:.6f}{self.colors.ENDC}\n"
        )
        if val_loss is not None:
            losses += (
                f"{self.colors.BOLD}Validation Metrics:{self.colors.ENDC}\n"
                f"├─ Current Loss: {self.colors.RED}{val_loss:.6f}{self.colors.ENDC}\n"
                f"├─ Best Loss:    {self.colors.GREEN}{self.best_metrics['val_loss']:.6f}{self.colors.ENDC}\n"
            )

        metrics_str = ""
        if metrics:
            metrics_str = (
                f"\n{self.colors.BOLD}Detailed Metrics:{self.colors.ENDC}\n"
                f"├─ RMSE: {self.colors.BLUE}{metrics['rmse']:.4f}{self.colors.ENDC} (Best: {self.colors.GREEN}{self.best_metrics['rmse']:.4f}{self.colors.ENDC})\n"
                f"├─ MAE:  {self.colors.BLUE}{metrics['mae']:.4f}{self.colors.ENDC} (Best: {self.colors.GREEN}{self.best_metrics['mae']:.4f}{self.colors.ENDC})\n"
                f"├─ MAPE: {self.colors.BLUE}{metrics['mape']:.2f}%{self.colors.ENDC}\n"
                f"└─ MASE: {self.colors.BLUE}{metrics['mase']:.4f}{self.colors.ENDC}"
            )

        timing = (
            f"\n{self.colors.BOLD}Timing Information:{self.colors.ENDC}\n"
            f"├─ Elapsed:        {self.colors.YELLOW}{str(timedelta(seconds=int(time_elapsed)))}{self.colors.ENDC}\n"
            f"├─ ETA:            {self.colors.GREEN}{str(eta)}{self.colors.ENDC}\n"
            f"└─ Time per epoch: {self.colors.BLUE}{time_per_epoch:.1f}s{self.colors.ENDC}"
        )

        footer = f"\n{self.separator}"

        # Combine all sections
        full_summary = f"{header}{losses}{metrics_str}{timing}{footer}"
        self.logger.info(full_summary)

    def calculate_metrics(self, y_pred: Optional[torch.Tensor], y_true: Optional[torch.Tensor]) -> dict:
        """Calculate metrics for the current epoch."""
        if y_pred is None or y_true is None:
            return {}
            
        try:
            dummy_timestamps = np.arange(len(y_pred))
            eval_result = self.evaluator.evaluate(y_pred, y_true, dummy_timestamps)
            return eval_result.total_metrics
        except Exception as e:
            self.logger.warning(f"Failed to calculate metrics: {str(e)}")
            return {}