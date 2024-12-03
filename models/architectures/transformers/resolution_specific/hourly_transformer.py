from typing import Dict, Any, Optional
import torch
from torch import nn, Tensor
import pandas as pd

from data_loading.types.interval_types import TimeInterval
from .base_resolution_transformer import BaseResolutionTransformer
from models.components.layers import EncoderLayer, DecoderLayer

class HourlyTransformer(BaseResolutionTransformer):
    """Transformer optimized for hourly predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        # Validate resolution before initialization
        if config.get('forecast_resolution_minutes', 60) > 180:
            raise ValueError("HourlyTransformer cannot handle resolutions > 180 minutes")
            
        super().__init__(config)
        
        # Hourly-specific configurations
        self.minutes_per_step = config.get('forecast_resolution_minutes', 60)
        
        # Enhanced pattern recognition for hourly data
        self.hourly_pattern_conv = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=4,  # Capture 4-hour patterns
            padding=2,
            groups=self.d_model
        )

    @classmethod
    def get_resolution_type(cls) -> TimeInterval:
        """Get the time interval type for hourly transformer."""
        return TimeInterval.HOURLY

    # ...rest of implementation similar to SubhourlyTransformer...