# models/interfaces.py
"""Core interfaces and types for the model system."""
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset

from training.reports.training_report import TrainingReport


class ModelInterface(ABC):
    """Base interface for all models."""
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def get_input_dims(self) -> int:
        pass

    @abstractmethod
    def get_output_dims(self) -> int:
        pass


class WrapperInterface(ABC):
    """Base interface for model wrappers."""
    @abstractmethod
    def train(
            self,
            train_dataset: Dataset[Any],
            validation_dataset: Optional[Dataset[Any]] = None
    ) -> TrainingReport:
        """Train the model."""
        pass

    @abstractmethod
    def training_step(
            self,
            batch_input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
            batch_target: torch.Tensor
    ) -> float:
        """Perform a single training step and return loss."""
        pass

    @abstractmethod
    def validation_step(
            self,
            batch_input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
            batch_target: torch.Tensor
    ) -> float:
        """Perform a single validation step and return loss."""
        pass

    @abstractmethod
    def predict(self, dataset: Dataset[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions using the model."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model state."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model state."""
        pass
