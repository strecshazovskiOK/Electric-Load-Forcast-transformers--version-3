# models/config/model_config.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from ..registry.model_types import ModelType

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_type: ModelType
    input_features: int
    output_features: int = 1
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 672  # Updated: 28 days * 24 hours * 4 intervals
    transformer_labels_count: int = 48  # Updated: 12 hours * 4 intervals
    points_per_interval: int = 4  # New field for 15-min intervals
    
    # Neural network specific
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    activation: str = 'relu'

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_epochs: int = 100
    optimizer: str = 'adam'
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    scheduler: Optional[str] = None
    scheduler_config: Dict[str, Any] = field(default_factory=dict)
    criterion: str = 'mse'
    criterion_config: Dict[str, Any] = field(default_factory=dict)

    # Device
    device: str = 'cuda'

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

def get_default_config(model_type: str) -> ModelConfig:
    """Get default configuration for specified model type."""
    base_config = {
        'model_type': model_type,
        'input_features': 1,
        'output_dim': 1
    }

    if model_type in {'vanilla_transformer', 'conv_transformer', 'informer'}:
        base_config.update({
            'd_model': 512,
            'n_heads': 8,
            'n_encoder_layers': 3,
            'n_decoder_layers': 3,
            'd_ff': 2048,
            'dropout': 0.1,
        })

    return ModelConfig(**base_config)