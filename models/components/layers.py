# models/components/layers.py
import torch
from torch import nn, Tensor
from typing import Optional, Callable, Union

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
            activation: Union[str, Callable] = "relu"
    ):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = {
                "relu": nn.ReLU(),
                "gelu": nn.GELU()
            }[activation.lower()]
        else:
            self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        return self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(x)
                )
            )
        )


class EncoderLayer(nn.Module):
    """Transformer encoder layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 attention_type: str = "standard", activation: str = "relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))


class DecoderLayer(nn.Module):
    """Transformer decoder layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 attention_type: str = "standard", activation: str = "relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        cross_attn_output, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        ff_output = self.feed_forward(x)
        return self.norm3(x + self.dropout(ff_output))
