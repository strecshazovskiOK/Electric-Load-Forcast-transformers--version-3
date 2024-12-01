# models/base/base_wrapper.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ..interfaces import WrapperInterface
from ..registry.model_types import ModelType

class BaseWrapper(WrapperInterface):
    """Base class for model wrappers providing consistent interface."""

    def __init__(self, model_type: ModelType, config: Dict[str, Any]):
        self.model_type = model_type
        self.config = config

    @abstractmethod
    def train(
            self,
            train_dataset: Dataset,
            validation_dataset: Optional[Dataset] = None
    ):
        pass

    @abstractmethod
    def predict(self, dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @classmethod
    def from_str(cls, model_str: str) -> 'BaseWrapper':
        """
        Creates a BaseWrapper instance from its string representation.
        Args:
            model_str: String representation of the model wrapper
        Returns:
            Reconstructed BaseWrapper instance
        """
        # Since model_str is the string representation, we need to parse it
        # Assuming format like "ModelType.TRANSFORMER:{'param1': value1, ...}"
        model_type_str, config_str = model_str.split(':', 1)
        model_type = ModelType[model_type_str.strip()]
        
        # Convert string config to dict (assuming it's in a valid format)
        config = eval(config_str)  # Note: In production, use ast.literal_eval or json.loads for safety
        
        return cls(model_type=model_type, config=config)

    def __str__(self) -> str:
        """String representation that can be parsed by from_str"""
        return f"{self.model_type}:{self.config}"