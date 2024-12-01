# models/base/base_model.py
from abc import abstractmethod
from typing import Any, Dict
import torch
from torch import nn

from ..interfaces import ModelInterface

class BaseModel(nn.Module, ModelInterface):
    """Base class for all models in the project."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the base model with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the model.
        """
        super().__init__() # type: ignore
        self.config = config

    def get_model_config(self) -> Dict[str, Any]:
        return self.config

    @abstractmethod
    def forward(self, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass of the model."""
        pass

    @abstractmethod
    def get_input_dims(self) -> int:
        """Get the input dimensions required by the model."""
        pass

    @abstractmethod
    def get_output_dims(self) -> int:
        """Get the output dimensions of the model."""
        pass


    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseModel':
        """Create a model instance from configuration."""
        return cls(config)
