from typing import Dict, Any, Optional
import torch
from torch import nn, Tensor
import pandas as pd

from data_loading.types.interval_types import TimeInterval
from .base_resolution_transformer import BaseResolutionTransformer
from models.components.layers import EncoderLayer, DecoderLayer

class MonthlyTransformer(BaseResolutionTransformer):
    """Transformer optimized for monthly predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Monthly-specific configurations
        self.minutes_per_step = config.get('forecast_resolution_minutes', 43200)  # 30 days default
        
        # Enhanced pattern recognition for monthly data
        self.monthly_pattern_conv = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=12,  # Capture yearly patterns
            padding=6,
            groups=self.d_model
        )

    @classmethod
    def get_resolution_type(cls) -> TimeInterval:
        """Get the time interval type for monthly transformer."""
        return TimeInterval.MONTHLY

    # ...rest of implementation similar to SubhourlyTransformer...