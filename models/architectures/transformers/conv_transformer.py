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
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False
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
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False
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
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract configuration
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.n_encoder_layers = config['n_encoder_layers']
        self.n_decoder_layers = config['n_decoder_layers']
        self.d_ff = config['d_ff']
        self.dropout = config.get('dropout', 0.1)
        self.input_features = config['input_features']
        self.kernel_size = config.get('kernel_size', 3)
        self.output_dim = 1  # We only want to predict energy consumption
        self.transformer_labels_count = config.get('transformer_labels_count', 48)
        
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

        # Output projection - only predict energy consumption
        self.projection = nn.Linear(self.d_model, self.output_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor, x_dec: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass supporting both single-input and encoder-decoder modes.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, features]
            x_dec: Optional decoder input tensor
        Returns:
            Output tensor of predictions [batch_size, seq_length, 1]
        """
        if x_dec is None:
            # Split input sequence for encoder and decoder
            split_point = x.size(1) - self.transformer_labels_count
            x_enc = x[:, :split_point, :]  # Keep all features for encoding
            x_dec = x[:, split_point:, :]  # Keep all features for decoding
        else:
            x_enc = x

        # Apply embeddings
        enc_embedding = self.encoder_embedding(x_enc)
        dec_embedding = self.decoder_embedding(x_dec)

        # Create appropriate mask
        tgt_mask = self.generate_square_subsequent_mask(dec_embedding.size(1)).to(x.device)

        # Run through encoder and decoder
        memory = self.encoder(enc_embedding)
        decoder_output = self.decoder(dec_embedding, memory, tgt_mask=tgt_mask)

        # Project to energy consumption prediction only
        output = self.projection(self.relu(decoder_output))

        return output  # Shape: [batch_size, seq_length, 1]

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generate causal mask for decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def get_input_dims(self) -> int:
        """Return the number of input features."""
        return self.input_features

    def get_output_dims(self) -> int:
        """Return the number of output dimensions."""
        return self.output_dim