# models/components/layers.py
import torch
from torch import nn, Tensor
from typing import Optional, Callable, Union

from models.components.activation import GELU, ReLU

from .attention import (
    MultiHeadAttention,
    ConvolutionalAttention,
    ProbSparseAttention
)


class FeedForwardNetwork(nn.Module):
    """Standard feed-forward network used in transformers."""

    def __init__(
            self,
            d_model: int,
            d_ff: int,
            dropout: float = 0.1,
            activation: str = "relu"
    ):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Use our new activation functions
        self.activation = GELU() if activation == "gelu" else ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(x)
                )
            )
        )



class EncoderLayer(nn.Module):
    """Transformer encoder layer."""
    
    def __init__(
        self, 
        d_model: int,
        n_heads: int, 
        d_ff: int,
        dropout: float,
        attention_type: str = "standard",
        kernel_size: int = 3,
        activation: str = "relu",
        batch_first: bool = True
    ):
        super().__init__()
        
        # Select attention mechanism
        if attention_type == "standard":
            self.self_attn = MultiHeadAttention(
                d_model, n_heads, dropout=dropout, batch_first=batch_first
            )
        elif attention_type == "convolutional":
            self.self_attn = ConvolutionalAttention(
                d_model, n_heads, kernel_size=kernel_size, dropout=dropout
            )
        elif attention_type == "prob_sparse":
            self.self_attn = ProbSparseAttention(
                d_model, n_heads, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        # Feed forward and normalization
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        # Self attention block
        attn_output, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False
        )
        src = self.norm1(src + self.dropout(attn_output))
        
        # Feed forward block
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))
        
        return src


class DecoderLayer(nn.Module):
    """Transformer decoder layer with improved implementation."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        attention_type: str = "standard",
        kernel_size: int = 3,
        activation: str = "relu",
        batch_first: bool = True
    ):
        super().__init__()
        
        # Select attention mechanisms
        if attention_type == "standard":
            self.self_attn = MultiHeadAttention(
                d_model, n_heads, dropout=dropout, batch_first=batch_first
            )
            self.cross_attn = MultiHeadAttention(
                d_model, n_heads, dropout=dropout, batch_first=batch_first
            )
        elif attention_type == "convolutional":
            self.self_attn = ConvolutionalAttention(
                d_model, n_heads, kernel_size=kernel_size, dropout=dropout
            )
            self.cross_attn = ConvolutionalAttention(
                d_model, n_heads, kernel_size=kernel_size, dropout=dropout
            )
        elif attention_type == "prob_sparse":
            self.self_attn = ProbSparseAttention(
                d_model, n_heads, dropout=dropout
            )
            self.cross_attn = ProbSparseAttention(
                d_model, n_heads, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        # Feed forward and normalization
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        # Self attention block
        self_attn_output, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False
        )
        tgt = self.norm1(tgt + self.dropout(self_attn_output))
        
        # Cross attention block
        cross_attn_output, _ = self.cross_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False
        )
        tgt = self.norm2(tgt + self.dropout(cross_attn_output))
        
        # Feed forward block
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_output))
        
        return tgt
