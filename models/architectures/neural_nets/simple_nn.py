# models/architectures/neural_nets/simple_nn.py
from typing import Dict, Any, List
import torch
from torch import nn

from models.base.base_model import BaseModel

from ...registry.factory import ModelFactory
from ...registry.model_types import ModelType


@ModelFactory.register(ModelType.SIMPLE_NEURAL_NET)
class SimpleNeuralNet(BaseModel):
    """Simple feed-forward neural network implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.input_dim = config['input_features']
        self.hidden_dims = config.get('hidden_dims', [64, 32])
        self.output_dim = config.get('output_dim', 1)
        self.dropout_rate = config.get('dropout', 0.1)
        self.activation = self._get_activation(config.get('activation', 'relu'))

        # Build network layers
        self.layers = self._build_network()

    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        activations: Dict[str, nn.Module] = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        if activation_name not in activations:
            raise ValueError(f"Unknown activation: {activation_name}")
        return activations[activation_name]

    def _build_network(self) -> nn.Sequential:
        """Build neural network layers."""
        layers: List[nn.Module] = []
        prev_dim = self.input_dim

        # Add hidden layers
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))

        return nn.Sequential(*layers)

    def forward(self, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            args: Input tensor of shape [batch_size, input_features]
            kwargs: Additional arguments

        Returns:
            Predictions tensor of shape [batch_size, output_dim]
        """
        x = args[0]  # Assuming the first positional argument is the input tensor
        return self.layers(x)

    def get_input_dims(self) -> int:
        return self.input_dim

    def get_output_dims(self) -> int:
        return self.output_dim

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default model configuration."""
        return {
            'input_features': 1,
            'hidden_dims': [64, 32],
            'output_dim': 1,
            'dropout': 0.1,
            'activation': 'relu'
        }