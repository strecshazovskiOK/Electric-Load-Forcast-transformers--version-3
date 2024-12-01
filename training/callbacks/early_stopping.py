# training/callbacks/early_stopping.py
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np

from .base_callback import TrainerCallback

class EarlyStopping(TrainerCallback):
    """Early stopping callback to prevent overfitting."""

    def __init__(self, patience: int, monitor: str = 'val_loss', min_delta: float = 0.0):
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.wait = 0
        self.best_value = np.inf
        self.stopped_epoch = 0
        self.best_weights: Optional[Dict[str, Any]] = None
        self.model: Optional[nn.Module] = None
        self._stop_training = False

    def on_training_begin(self, model: nn.Module, config: Dict[str, Any]) -> None:
        self.model = model
        self.wait = 0
        self.best_value = np.inf
        self.stopped_epoch = 0
        self.best_weights = None
        self._stop_training = False

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        current = logs.get(self.monitor)
        if current is None or self.model is None:
            return

        if current - self.min_delta < self.best_value:
            self.best_value = current
            self.wait = 0
            # Save a copy of the model state
            self.best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience and self.best_weights is not None:
                self.stopped_epoch = epoch
                # Load the best weights back
                self.model.load_state_dict(self.best_weights)
                self._stop_training = True

    def on_training_end(self, model: nn.Module, config: Dict[str, Any]) -> None:
        if self.stopped_epoch > 0:
            print(f'Early stopping occurred at epoch {self.stopped_epoch}')

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        pass

    @property
    def stop_training(self) -> bool:
        return self._stop_training