from typing import Dict, Any, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer

from models.registry.factory import ModelFactory
from models.registry.model_types import ModelType
from models.components.embeddings import TotalEmbedding
from models.components.attention import MultiHeadAttention
from models.base.base_model import BaseModel


class ConvEncoderLayer(TransformerEncoderLayer):
    """Transformer encoder layer with convolutional attention."""
    
    def __init__(
        self, 
        d_model: int,
        n_heads: int,
        d_ff: int,
        kernel_size: int,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = True
    ):
        super().__init__(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )
        
        # Replace standard attention with convolutional attention
        self.self_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        
        # Add convolutional layers
        padding = (kernel_size - 1) // 2
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size, padding=padding)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size, padding=padding)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        # Apply convolutions to queries and keys
        q = self.conv_q(src.transpose(1, 2)).transpose(1, 2)
        k = self.conv_k(src.transpose(1, 2)).transpose(1, 2)
        
        # Self attention block with convolutional features
        src2, _ = self.self_attn(q, k, src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class ConvDecoderLayer(TransformerDecoderLayer):
    """Transformer decoder layer with convolutional attention."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        kernel_size: int,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = True
    ):
        super().__init__(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )
        
        # Replace standard attention with convolutional attention
        self.self_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        self.multihead_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        
        # Add convolutional layers
        padding = (kernel_size - 1) // 2
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size, padding=padding)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size, padding=padding)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        # Apply convolutions
        q_tgt = self.conv_q(tgt.transpose(1, 2)).transpose(1, 2)
        k_tgt = self.conv_k(tgt.transpose(1, 2)).transpose(1, 2)
        q_mem = self.conv_q(memory.transpose(1, 2)).transpose(1, 2)
        
        # Self attention
        tgt2, _ = self.self_attn(q_tgt, k_tgt, tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention
        tgt2, _ = self.multihead_attn(tgt, q_mem, memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


@ModelFactory.register(ModelType.CONV_TRANSFORMER)
class ConvolutionalTransformer(BaseModel):
    """Transformer with convolutional attention mechanism."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)  # Fix: Pass config to parent class
        
        # Extract configuration
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.n_encoder_layers = config['n_encoder_layers']
        self.n_decoder_layers = config['n_decoder_layers']
        self.d_ff = config['d_ff']
        self.dropout = config.get('dropout', 0.1)
        self.input_features = config['input_features']
        self.kernel_size = config.get('kernel_size', 3)
        
        # Create encoder
        encoder_layer = ConvEncoderLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        )
        encoder_norm = nn.LayerNorm(self.d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.n_encoder_layers, encoder_norm)
        
        # Create decoder
        decoder_layer = ConvDecoderLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        )
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.n_decoder_layers, decoder_norm)

        # Initialize embeddings
        self.encoder_embedding = TotalEmbedding(
            d_model=self.d_model,
            value_features=1,
            time_features=self.input_features - 1,
            dropout=self.dropout
        )
        self.decoder_embedding = TotalEmbedding(
            d_model=self.d_model,
            value_features=1,
            time_features=self.input_features - 1,
            dropout=self.dropout
        )

        # Output projection
        self.projection = nn.Linear(self.d_model, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(
        self,
        x_enc: Tensor,
        x_dec: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass with convolutional attention.

        Args:
            x_enc: Encoder input [batch_size, seq_enc_length, features]
            x_dec: Decoder input [batch_size, seq_dec_length, features]
            src_mask: Optional mask for encoder
            tgt_mask: Optional mask for decoder
        """
        # Apply embeddings
        enc_embedding = self.encoder_embedding(x_enc)
        dec_embedding = self.decoder_embedding(x_dec)

        # Run through encoder and decoder
        memory = self.encoder(enc_embedding, mask=src_mask)
        output = self.decoder(dec_embedding, memory, tgt_mask=tgt_mask)

        # Project to output dimension
        return self.projection(self.relu(output))

    def create_masks(self, src_len: int, tgt_len: int) -> Tuple[Optional[Tensor], Tensor]:
        """Create appropriate masks for training."""
        src_mask = None
        tgt_mask = self.generate_square_subsequent_mask(tgt_len)
        return src_mask, tgt_mask

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generate causal mask for decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask