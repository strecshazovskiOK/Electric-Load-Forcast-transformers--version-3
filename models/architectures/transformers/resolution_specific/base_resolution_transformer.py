# models/architectures/transformers/resolution_specific/base_resolution_transformer.py
from typing import Dict, Any, Optional
import torch
from torch import nn, Tensor

from data_loading.types.interval_types import TimeInterval
from utils.logging.logger import Logger
from utils.logging.config import LoggerConfig, LogLevel

from ..base_transformer import BaseTransformer

class BaseResolutionTransformer(BaseTransformer):
    """Base class for all resolution-specific transformer implementations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the resolution-aware transformer."""
        # Call parent initialization first
        super().__init__(config)
        
        self.logger = Logger.get_logger(__name__)
        self.logger.debug("BaseResolutionTransformer initialization")
        
        # Resolution-specific configuration
        self.input_resolution = config['input_resolution_minutes']
        self.forecast_resolution = config['forecast_resolution_minutes']
        self.minutes_per_step = self.input_resolution
        
        # Initialize resampling if needed
        self.resampling_needed = self.input_resolution != self.forecast_resolution
        if self.resampling_needed:
            self._setup_resampling_layer()

    def _setup_resampling_layer(self) -> None:
        """Setup resampling layer if input and forecast resolutions differ."""
        if self.resampling_needed:
            # If downsampling (e.g., 15min -> 1hour), use average pooling
            if self.resampling_factor > 1:
                pool_size = int(self.resampling_factor)
                self.resampling_layer = nn.AvgPool1d(
                    kernel_size=pool_size,
                    stride=pool_size
                )
            # If upsampling (rare case), use linear interpolation
            else:
                self.resampling_layer = nn.Upsample(
                    scale_factor=1/self.resampling_factor,
                    mode='linear'
                )

    def _resample_if_needed(self, x: Tensor) -> Tensor:
        """Resample input data if resolutions differ."""
        if not self.resampling_needed:
            return x
            
        # Reshape for resampling [batch, seq, features] -> [batch, features, seq]
        x = x.transpose(1, 2)
        
        # Apply resampling
        x = self.resampling_layer(x)
        
        # Reshape back [batch, features, seq] -> [batch, seq, features]
        return x.transpose(1, 2)

    def _get_positional_encoding(self, sequence_length: int) -> Tensor:
        """Get positional encoding appropriate for the resolution."""
        # Override in specific implementations if needed
        return super().generate_square_subsequent_mask(sequence_length)

    def _adjust_attention_for_resolution(
        self,
        attention_weights: Tensor,
        resolution_minutes: int
    ) -> Tensor:
        """Adjust attention weights based on resolution."""
        # For longer resolutions, we might want to enforce longer-term dependencies
        if resolution_minutes >= 1440:  # daily or longer
            # Enhance attention to seasonal patterns
            attention_weights = attention_weights * (1 + torch.cos(
                torch.arange(attention_weights.size(-1)) * 2 * torch.pi / (7 * 24 * 60 / resolution_minutes)
            ).to(attention_weights.device))
        
        return attention_weights

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with resolution handling."""
        # Resample input if needed
        if self.resampling_needed:
            src = self._resample_if_needed(src)
            tgt = self._resample_if_needed(tgt)

        # Get resolution-appropriate masks if none provided
        if src_mask is None:
            src_mask = self._get_positional_encoding(src.size(1))
        if tgt_mask is None:
            tgt_mask = self._get_positional_encoding(tgt.size(1))

        return super().forward(
            src,
            tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

    @classmethod
    def get_resolution_type(cls) -> TimeInterval:
        """Get the time interval type this transformer is designed for."""
        raise NotImplementedError(
            "Resolution-specific transformers must implement get_resolution_type"
        )

    def _validate_resolution(self) -> None:
        """Validate that the configured resolution matches the transformer type."""
        expected_type = self.get_resolution_type()
        if self.forecast_resolution > expected_type.value:
            raise ValueError(
                f"Forecast resolution ({self.forecast_resolution} minutes) exceeds "
                f"maximum for {expected_type.name} transformer ({expected_type.value} minutes)"
            )