from typing import Dict, Any, Optional
import torch
from torch import nn, Tensor
import pandas as pd

from data_loading.types.interval_types import TimeInterval
from .base_resolution_transformer import BaseResolutionTransformer
from models.components.layers import EncoderLayer, DecoderLayer

class SubhourlyTransformer(BaseResolutionTransformer):
    """Transformer optimized for sub-hourly predictions (≤60 minutes)."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SubhourlyTransformer with proper layer creation.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        # Ensure output_features is in config
        if 'output_features' not in config:
            config['output_features'] = 1  # Default to 1 if not specified
        
        # Call parent class initialization first
        super().__init__(config)
        
        # Store configuration parameters
        self.n_encoder_layers = config.get('n_encoder_layers', 6)
        self.n_decoder_layers = config.get('n_decoder_layers', 6)
        self.d_model = config.get('d_model', 512)
        self.n_heads = config.get('n_heads', 8)
        self.d_ff = config.get('d_ff', 2048)
        self.dropout = config.get('dropout', 0.1)
        
        # Calculate steps per hour based on resolution
        self.minutes_per_step = config.get('forecast_resolution_minutes', 15)
        self.steps_per_hour = 60 // self.minutes_per_step
        
        # Create encoder layers
        self.encoder_layers = self._create_encoder_layers()
        
        # Create decoder layers
        self.decoder_layers = self._create_decoder_layers()
        
        # Initialize short-term pattern recognition
        self.short_term_conv = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=3,
            padding=1,
            groups=self.d_model  # Depthwise convolution for efficiency
        )
        
        print("DEBUG: SubhourlyTransformer initialization complete")
        print(f"DEBUG: Configuration loaded - n_encoder_layers: {self.n_encoder_layers}, d_model: {self.d_model}")

    def _create_encoder_layers(self) -> nn.ModuleList:
        """Create encoder layers optimized for sub-hourly patterns."""
        return nn.ModuleList([
            EncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                attention_type="standard",  # Use standard attention for short sequences
                activation="gelu"  # GELU for better gradient flow
            ) for _ in range(self.n_encoder_layers)
        ])

    def _create_decoder_layers(self) -> nn.ModuleList:
        """Create decoder layers optimized for sub-hourly patterns."""
        return nn.ModuleList([
            DecoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                attention_type="standard",
                activation="gelu"
            ) for _ in range(self.n_decoder_layers)
        ])

    def encode(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Enhanced encoding with short-term pattern recognition."""
        # Standard embedding
        src = self.encoder_embedding(src)
        
        # Apply short-term pattern recognition
        src_conv = src.transpose(1, 2)  # [batch, d_model, seq_len]
        src_conv = self.short_term_conv(src_conv)
        src_conv = src_conv.transpose(1, 2)  # [batch, seq_len, d_model]
        
        # Combine with original embeddings
        src = src + src_conv
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            src = layer(
                src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask
            )
        return src

    def _adjust_attention_for_resolution(
        self,
        attention_weights: Tensor,
        resolution_minutes: int
    ) -> Tensor:
        """Adjust attention weights for sub-hourly patterns."""
        # Enhance attention to recent time steps
        sequence_length = attention_weights.size(-1)
        recent_steps = min(self.steps_per_hour * 2, sequence_length)  # Last 2 hours
        
        # Create decaying weights for recent time steps
        decay = torch.exp(torch.arange(recent_steps, 0, -1, device=attention_weights.device) * -0.1)
        decay = torch.cat([torch.ones(sequence_length - recent_steps, device=attention_weights.device), decay])
        
        # Apply decay to attention weights
        return attention_weights * decay.unsqueeze(0).unsqueeze(0)

    @classmethod
    def get_resolution_type(cls) -> TimeInterval:
        """Get the time interval type for sub-hourly transformer."""
        return TimeInterval.FIFTEEN_MIN

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with enhanced sub-hourly pattern recognition."""
        output = super().forward(
            src, tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask= tgt_key_padding_mask
        )
        
        # Additional processing for sub-hourly predictions if needed
        if hasattr(self, 'output_processing'):
            output = self.output_processing(output)
            
        return output

    def _handle_temporal_features(
        self,
        x: Tensor,
        timestamps: torch.Tensor
    ) -> Tensor:
        """Handle temporal features for sub-hourly data."""
        # Convert tensor timestamps to pandas datetime
        dates = pd.to_datetime(timestamps.cpu().numpy(), unit='s')
        
        # Extract temporal features
        minutes = torch.tensor(dates.minute.values, device=timestamps.device)
        hours = torch.tensor(dates.hour.values, device=timestamps.device)
        
        # Process and return features
        return x  # This should be implemented with actual feature processing