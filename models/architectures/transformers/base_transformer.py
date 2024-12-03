from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import torch
from torch import nn, Tensor

from models.base.base_model import BaseModel
from models.components.embeddings import CombinedEmbedding
from models.components.attention import MultiHeadAttention
from utils.logging.logger import Logger
from utils.logging.config import LoggerConfig, LogLevel

class BaseTransformer(BaseModel, ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize logger
        logger_config = LoggerConfig(
            level=LogLevel.INFO,
            component_name="BaseTransformer",
            include_timestamp=True
        )
        self.logger = Logger.get_logger(__name__, logger_config)
        self.logger.debug("Initializing BaseTransformer", {"config": config})

        # Validate configuration
        required_keys = ['d_model', 'n_heads', 'input_features']
        if missing_keys := [key for key in required_keys if key not in config]:
            self.logger.error("Missing required configuration keys", {"missing_keys": missing_keys})
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

        self.d_model = config['d_model']
        self.input_features = config['input_features']
        self.output_features = config.get('output_features', 1)
        
        self.logger.debug("Model dimensions initialized", {
            "d_model": self.d_model,
            "input_features": self.input_features,
            "output_features": self.output_features
        })

        # Initialize embeddings
        self.logger.debug("Creating encoder embedding")
        self.encoder_embedding = CombinedEmbedding(
            d_model=self.d_model,
            input_features=self.input_features,
            dropout=config['dropout']
        )

        self.logger.debug("Creating decoder embedding")
        self.decoder_embedding = CombinedEmbedding(
            d_model=self.d_model,
            input_features=self.input_features,
            dropout=config['dropout']
        )

        # Initialize output projection
        self.output_projection = nn.Linear(self.d_model, self.output_features)
        self.logger.debug("Initialized output projection layer")
        
        # Initialize output projection weights
        self._initialize_output_projection()

        self.debug_counter = 0
        self.max_debug_prints = 2

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        required_keys = ['d_model', 'n_heads', 'input_features']
        if missing_keys := [key for key in required_keys if key not in config]:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

        if config['d_model'] % config['n_heads'] != 0:
            raise ValueError(
                f"d_model ({config['d_model']}) must be divisible by "
                f"n_heads ({config['n_heads']})"
            )

    def _create_embedding(self) -> CombinedEmbedding:
        """Create embedding layer with proper initialization."""
        return CombinedEmbedding(
            d_model=self.d_model,
            input_features=self.input_features,
            dropout=self.dropout
        )

    def _initialize_output_projection(self) -> None:
        """Initialize output projection layer weights."""
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    @abstractmethod
    def _create_encoder_layers(self) -> nn.ModuleList:
        """Create encoder layers specific to the transformer variant."""
        pass

    @abstractmethod
    def _create_decoder_layers(self) -> nn.ModuleList:
        """Create decoder layers specific to the transformer variant."""
        pass

    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None,
              src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Encode input sequence."""
        if self.debug_counter < self.max_debug_prints:
            self.logger.debug(f"Encoder device status (call {self.debug_counter + 1}/{self.max_debug_prints})", {
                "source_device": str(src.device),
                "source_mask_device": str(src_mask.device) if src_mask is not None else "None",
                "model_device": str(next(self.parameters()).device)
            })
            self.debug_counter += 1
        
        # Move masks to correct device
        if src_mask is not None:
            src_mask = src_mask.to(src.device)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(src.device)

        # Apply embedding
        src = self.encoder_embedding(src)
        
        # Pass through encoder layers
        for i, layer in enumerate(self.encoder_layers):
            self.logger.debug("Encoder layer processing", {
                "layer": i,
                "input_device": str(src.device)
            })
            src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return src

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Decode target sequence."""
        # Apply embedding
        tgt = self.decoder_embedding(tgt)
        
        # Create or adjust masks to match sequence dimensions
        if memory_mask is not None:
            # Adjust memory mask shape to match [tgt_len, src_len]
            memory_mask = self._adjust_memory_mask(memory_mask, tgt.size(1), memory.size(1))

        # Pass through decoder layers
        for layer in self.decoder_layers:
            tgt = layer(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        return tgt

    def _adjust_memory_mask(self, mask: Tensor, tgt_len: int, src_len: int) -> Tensor:
        """Adjust memory mask to match required dimensions."""
        if mask.size(0) != tgt_len or mask.size(1) != src_len:
            self.logger.debug("Adjusting memory mask dimensions", {
                "original_shape": tuple(mask.shape),
                "new_shape": (tgt_len, src_len)
            })
            # Create new mask with correct dimensions
            new_mask = torch.zeros((tgt_len, src_len), device=mask.device)
            # Copy values where possible
            min_rows = min(mask.size(0), tgt_len)
            min_cols = min(mask.size(1), src_len)
            new_mask[:min_rows, :min_cols] = mask[:min_rows, :min_cols]
            return new_mask
        return mask

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with correct mask handling."""
        if self.debug_counter < self.max_debug_prints:
            self.logger.debug(f"Forward pass details (call {self.debug_counter + 1}/{self.max_debug_prints})", {
                "source_shape": tuple(src.shape),
                "target_shape": tuple(tgt.shape),
                "source_mask_shape": tuple(src_mask.shape) if src_mask is not None else None,
                "target_mask_shape": tuple(tgt_mask.shape) if tgt_mask is not None else None
            })
        
        # Encode source sequence
        memory = self.encode(src, src_mask, src_key_padding_mask)
        
        # Decode target sequence with adjusted masks
        output = self.decode(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,  # This will be adjusted in decode()
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # Project to output dimension
        output = self.output_projection(output)
        
        return output

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        # Move mask to same device as model
        return mask.to(next(self.parameters()).device)
    
    def get_input_dims(self) -> int:
        return self.input_features

    def get_output_dims(self) -> int:
        return 1