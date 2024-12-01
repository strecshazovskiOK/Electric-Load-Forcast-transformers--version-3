# training/callbacks/learning_rate_scheduler.py
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from .base_callback import TrainerCallback

class LRSchedulerCallback(TrainerCallback):
    """Learning rate scheduler callback."""

    def __init__(self, scheduler: LRScheduler, monitor: str = 'val_loss'):
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(logs.get(self.monitor), epoch=epoch) # type: ignore
        else:
            self.scheduler.step()

    def on_training_begin(self, model: nn.Module, config: Dict[str, Any]) -> None:
        pass

    def on_training_end(self, model: nn.Module, config: Dict[str, Any]) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        pass