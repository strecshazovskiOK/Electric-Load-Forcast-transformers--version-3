from typing import Callable, Dict, Any, Optional, cast
import torch
from torch import nn, Tensor
import pandas as pd

from data_loading.types.interval_types import TimeInterval
from .base_resolution_transformer import BaseResolutionTransformer
from models.components.layers import EncoderLayer, DecoderLayer
from torch.utils.checkpoint import checkpoint

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
        self.use_checkpointing = config.get('use_checkpointing', True)  # Add explicit flag
        
        # Calculate steps per hour based on resolution
        self.minutes_per_step = config.get('forecast_resolution_minutes', 15)
        self.steps_per_hour = 60 // self.minutes_per_step
        
        # Create encoder and decoder layers
        self.encoder_layers = self._create_encoder_layers()
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

    def _create_custom_forward(self, layer: nn.Module) -> Callable[..., Tensor]:
        """Creates a type-safe forward function for checkpointing.
        
        This helper method ensures proper type handling for the checkpointed function.
        The returned function is guaranteed to return a Tensor, maintaining type safety.
        
        Args:
            layer: The encoder layer to be wrapped
            
        Returns:
            A callable that properly handles types for checkpointing
        """
        def custom_forward(*inputs: Any) -> Tensor:
            # Validate inputs
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise ValueError("First input must be a tensor")
            
            x = inputs[0]
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
                
            # Handle optional mask inputs
            mask = inputs[1] if len(inputs) > 1 else None
            key_padding_mask = inputs[2] if len(inputs) > 2 else None
            
            # Process through layer and ensure tensor output
            output = layer(x, src_mask=mask, src_key_padding_mask=key_padding_mask)
            if not isinstance(output, torch.Tensor):
                raise ValueError("Layer output must be a tensor")
                
            return output
            
        return custom_forward

    def encode(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
        ) -> Tensor:
            """Enhanced encoding with proper gradient handling."""
            
            # Initial gradient setup with function preservation
            if not src.requires_grad:
                src = src.clone().detach().requires_grad_(True)
                
            # Apply embedding while preserving gradients
            src = self.encoder_embedding(src)
            
            # Apply short-term pattern recognition with gradient preservation
            src_conv = src.transpose(1, 2)
            src_conv = self.short_term_conv(src_conv)
            src_conv = src_conv.transpose(1, 2)
            src = src + src_conv
            
            # Process through encoder layers with gradient-aware checkpointing
            for i, layer in enumerate(self.encoder_layers):
                if self.use_checkpointing and self.training:
                    def create_custom_forward():
                        def custom_forward(*inputs):
                            x = inputs[0]
                            if not x.requires_grad:
                                x = x.clone().detach().requires_grad_(True)
                                
                            mask = inputs[1] if len(inputs) > 1 else None
                            key_padding_mask = inputs[2] if len(inputs) > 2 else None
                            
                            # Run the layer computation
                            result = layer(x, src_mask=mask, src_key_padding_mask=key_padding_mask)
                            
                            # Ensure we preserve the gradient function
                            if not result.grad_fn:
                                result = result.clone()
                                
                            return result
                        return custom_forward

                    # Create fresh tensor copies to ensure proper gradient tracking
                    checkpoint_inputs = [
                        src.clone() if isinstance(src, Tensor) else src,
                        src_mask.clone() if isinstance(src_mask, Tensor) else src_mask,
                        src_key_padding_mask.clone() if isinstance(src_key_padding_mask, Tensor) else src_key_padding_mask
                    ]
                    
                    src = checkpoint(
                        create_custom_forward(),
                        *checkpoint_inputs,
                        use_reentrant=False,
                        preserve_rng_state=True
                    ) # type: ignore
                    
                    # Ensure gradient tracking is maintained
                    if not src.grad_fn:
                        src = src.clone()
                else:
                    src = layer(
                        src,
                        src_mask=src_mask,
                        src_key_padding_mask=src_key_padding_mask
                    )
                
                # Debug logging for gradient tracking
                # print(f"Debug - Layer {i} output requires_grad: {src.requires_grad}")
                # print(f"Debug - Layer {i} grad_fn: {type(src.grad_fn).__name__ if src.grad_fn else 'None'}")
                # print(f"Debug - Layer {i} is_leaf: {src.is_leaf}")
                
            return src

    def _adjust_attention_for_resolution(
        self,
        attention_weights: Tensor,
        resolution_minutes: int
    ) -> Tensor:
        """Adjust attention weights for sub-hourly patterns with gradient tracking."""
        sequence_length = attention_weights.size(-1)
        recent_steps = min(self.steps_per_hour * 2, sequence_length)
        
        # Create decaying weights with gradients
        decay = torch.exp(torch.arange(
            recent_steps, 0, -1, 
            device=attention_weights.device, 
            dtype=attention_weights.dtype
        ) * -0.1).requires_grad_(True)
        
        # Concatenate with ones tensor
        ones = torch.ones(
            sequence_length - recent_steps,
            device=attention_weights.device,
            dtype=attention_weights.dtype,
            requires_grad=True
        )
        decay = torch.cat([ones, decay])
        
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
        """Forward pass with gradient tracking verification."""
        # Ensure input tensors have gradients enabled
        if not src.requires_grad:
            src = src.detach().requires_grad_(True)
        if not tgt.requires_grad:
            tgt = tgt.detach().requires_grad_(True)
            
        # Log gradient status for debugging
        # if torch.is_grad_enabled():
        #     print(f"Debug - Forward input src requires_grad: {src.requires_grad}")
        #     print(f"Debug - Forward input tgt requires_grad: {tgt.requires_grad}")
            
        output = super().forward(
            src, tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Verify gradient tracking is maintained
        # if torch.is_grad_enabled():
        #     print(f"Debug - Forward output requires_grad: {output.requires_grad}")
            
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