# training/callbacks/base_callback.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch.nn as nn

class TrainerCallback(ABC):
    """Base class for all callbacks."""

    @abstractmethod
    def on_training_begin(self, model: nn.Module, config: Dict[str, Any]) -> None:
        """Called at the beginning of training."""
        pass

    @abstractmethod
    def on_training_end(self, model: nn.Module, config: Dict[str, Any]) -> None:
        """Called at the end of training."""
        pass

    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Called at the beginning of an epoch."""
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Called at the end of an epoch."""
        pass