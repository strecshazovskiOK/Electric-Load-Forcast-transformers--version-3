from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import torch
from torch import nn, Tensor

from models.base.base_model import BaseModel
from models.components.embeddings import CombinedEmbedding
from models.components.attention import MultiHeadAttention


class BaseTransformer(BaseModel, ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.n_encoder_layers = config['n_encoder_layers']
        self.n_decoder_layers = config['n_decoder_layers']
        self.d_ff = config['d_ff']
        self.dropout = config['dropout']
        self.input_features = config['input_features']
        self.batch_first = config.get('batch_first', True)  # Default to batch_first=True

        # Common components with proper initialization
        self.encoder_embedding = CombinedEmbedding(
            self.d_model,
            self.input_features,
            dropout=self.dropout
        )
        self.decoder_embedding = CombinedEmbedding(
            self.d_model,
            self.input_features,
            dropout=self.dropout
        )

        # Initialize encoder and decoder layers
        self.encoder_layers = self._create_encoder_layers()
        self.decoder_layers = self._create_decoder_layers()

        # Output projection with proper initialization
        self.output_projection = nn.Linear(self.d_model, 1)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @abstractmethod
    def _create_encoder_layers(self) -> nn.ModuleList:
        """Create encoder layers specific to the transformer variant."""
        pass

    @abstractmethod
    def _create_decoder_layers(self) -> nn.ModuleList:
        """Create decoder layers specific to the transformer variant."""
        pass

    def encode(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode input sequence."""
        # Apply embedding
        src = self.encoder_embedding(src)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            src = layer(
                src,
                src_mask=src_mask, 
                src_key_padding_mask=src_key_padding_mask
            )
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

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass that returns only the final output."""
        # Encode source sequence
        memory = self.encode(src, src_mask, src_key_padding_mask)
        
        # Decode target sequence
        output = self.decode(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # Project to output dimension
        output = self.output_projection(output)
        
        return output

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generate a square mask for the sequence. 
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def get_input_dims(self) -> int:
        return self.input_features

    def get_output_dims(self) -> int:
        return 1