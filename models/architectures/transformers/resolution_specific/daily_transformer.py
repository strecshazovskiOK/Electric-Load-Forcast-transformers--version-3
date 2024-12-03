# models/architectures/transformers/resolution_specific/daily_transformer.py
from typing import Dict, Any, Optional
import torch
from torch import nn, Tensor
import pandas as pd

from data_loading.types.interval_types import TimeInterval
from .base_resolution_transformer import BaseResolutionTransformer
from models.components.layers import EncoderLayer, DecoderLayer

class DailyTransformer(BaseResolutionTransformer):
    """Transformer optimized for daily predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        # Validate resolution before initialization
        if config.get('forecast_resolution_minutes', 1440) > 2880:
            raise ValueError("DailyTransformer cannot handle resolutions > 2 days")
            
        super().__init__(config)
        
        # Daily-specific configurations
        self.minutes_per_step = config.get('forecast_resolution_minutes', 1440)
        
        # Enhanced pattern recognition for daily data
        self.daily_pattern_conv = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=7,  # Capture weekly patterns
            padding=3,
            groups=self.d_model
        )

    @classmethod
    def get_resolution_type(cls) -> TimeInterval:
        """Get the time interval type for daily transformer."""
        return TimeInterval.DAILY

    # ...rest of implementation similar to SubhourlyTransformer...