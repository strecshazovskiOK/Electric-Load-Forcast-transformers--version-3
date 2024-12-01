# models/registry/model_types.py
from enum import Enum, auto

class ModelType(Enum):  # Fixed syntax error
    """Enumeration of available model types."""

    TIME_SERIES_TRANSFORMER = auto()
    LINEAR_REGRESSION = auto()
    SIMPLE_NEURAL_NET = auto()
    VANILLA_TRANSFORMER = auto()
    CONV_TRANSFORMER = auto()
    INFORMER = auto()

    @property
    def is_transformer(self) -> bool:
        """Check if model type is a transformer variant."""
        return self in {
            self.VANILLA_TRANSFORMER,
            self.CONV_TRANSFORMER,
            self.INFORMER
        }

    @property
    def is_neural_net(self) -> bool:
        """Check if model type is a neural network."""
        return self in {
            self.SIMPLE_NEURAL_NET,
            self.VANILLA_TRANSFORMER,
            self.CONV_TRANSFORMER,
            self.INFORMER
        }

def initialize_model_registry():
    """Initialize the model registry by importing all model implementations."""
    from models.architectures.transformers.vanilla_transformer import VanillaTransformer
    # Add other model imports here as needed