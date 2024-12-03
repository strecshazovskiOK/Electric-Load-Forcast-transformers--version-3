# models/registry/model_types.py
from enum import Enum, auto
from typing import Optional

class ModelType(Enum):
    """Enumeration of available model types."""
    
    # Base model types
    LINEAR_REGRESSION = auto()
    SIMPLE_NEURAL_NET = auto()
    
    # Standard transformer variants
    VANILLA_TRANSFORMER = auto()
    CONV_TRANSFORMER = auto()
    INFORMER = auto()
    TIME_SERIES_TRANSFORMER = auto()
    
    # Resolution-specific transformers
    SUBHOURLY_TRANSFORMER = auto()
    HOURLY_TRANSFORMER = auto()
    DAILY_TRANSFORMER = auto()
    MONTHLY_TRANSFORMER = auto()

    @property
    def is_transformer(self) -> bool:
        """Check if model type is a transformer variant."""
        return self in {
            self.TIME_SERIES_TRANSFORMER,
            self.VANILLA_TRANSFORMER,
            self.CONV_TRANSFORMER,
            self.INFORMER,
            self.SUBHOURLY_TRANSFORMER,
            self.HOURLY_TRANSFORMER,
            self.DAILY_TRANSFORMER,
            self.MONTHLY_TRANSFORMER
        }

    @property
    def is_neural_net(self) -> bool:
        """Check if model type is a neural network."""
        return self.is_transformer or self == self.SIMPLE_NEURAL_NET

    @property
    def is_resolution_specific(self) -> bool:
        """Check if model type is resolution-specific."""
        return self in {
            self.SUBHOURLY_TRANSFORMER,
            self.HOURLY_TRANSFORMER,
            self.DAILY_TRANSFORMER,
            self.MONTHLY_TRANSFORMER
        }

    @classmethod
    def get_for_resolution(cls, resolution_minutes: int) -> 'ModelType':
        """Get appropriate transformer type for given resolution."""
        if resolution_minutes <= 15:
            return cls.SUBHOURLY_TRANSFORMER
        elif resolution_minutes <= 60:
            return cls.HOURLY_TRANSFORMER
        elif resolution_minutes <= 1440:
            return cls.DAILY_TRANSFORMER
        else:
            return cls.MONTHLY_TRANSFORMER

    @classmethod
    def from_string(cls, name: str) -> 'ModelType':
        """Convert string to ModelType."""
        try:
            return cls[name]
        except KeyError:
            raise ValueError(f"Unknown model type: {name}")

    def __str__(self) -> str:
        """String representation of the enum."""
        return self.name

def initialize_model_registry():
    """Initialize the model registry by importing all model implementations."""
    # Standard models
    from models.architectures.linear.linear_regression import LinearRegression
    from models.architectures.neural_nets.simple_nn import SimpleNeuralNet
    
    # Standard transformers
    from models.architectures.transformers.vanilla_transformer import VanillaTransformer
    from models.architectures.transformers.timeseries_transformer import TimeSeriesTransformer
    from models.architectures.transformers.conv_transformer import ConvolutionalTransformer
    from models.architectures.transformers.informer import Informer
    
    # Resolution-specific transformers
    from models.architectures.transformers.resolution_specific import (
        SubhourlyTransformer,
        HourlyTransformer,
        DailyTransformer,
        MonthlyTransformer
    )