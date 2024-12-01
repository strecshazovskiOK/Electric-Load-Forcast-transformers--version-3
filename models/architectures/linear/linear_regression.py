# models/architectures/linear/linear_regression.py
from typing import Dict, Any
import torch
from torch import nn

from ...base.base_model import BaseModel
from ...registry.factory import ModelFactory
from ...registry.model_types import ModelType


@ModelFactory.register(ModelType.LINEAR_REGRESSION)
class LinearRegression(BaseModel):
    """Linear regression model implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.input_dim = config['input_features']
        self.output_dim = config.get('output_dim', 1)

        self.linear = nn.Linear(
            self.input_dim,
            self.output_dim
        )

        # Optional bias initialization
        if config.get('zero_init_bias', True):
            nn.init.zeros_(self.linear.bias)

    def forward(self, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass of the linear model.

        Args:
            args: Input tensor of shape [batch_size, input_features]
            kwargs: Additional arguments

        Returns:
            Predictions tensor of shape [batch_size, output_dim]
        """
        x = args[0]  # Assuming the first positional argument is the input tensor
        return self.linear(x)

    def get_input_dims(self) -> int:
        return self.input_dim

    def get_output_dims(self) -> int:
        return self.output_dim

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default model configuration."""
        return {
            'input_features': 1,
            'output_dim': 1,
            'zero_init_bias': True
        }