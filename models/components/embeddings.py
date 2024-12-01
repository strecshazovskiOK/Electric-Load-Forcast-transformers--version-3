# models/components/embeddings.py
import math
import torch
from torch import nn, Tensor
from typing import Optional

class ValueEmbedding(nn.Module):
    """Projects time series values to model dimension."""

    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        self.projection = nn.Linear(n_features, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch_size, seq_len, features]
        return self.projection(x)

class PositionalEmbedding(nn.Module):
    """Adds positional information to embeddings."""

    def __init__(
            self,
            d_model: int,
            max_seq_len: int = 5000,
            dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CombinedEmbedding(nn.Module):
    """Combines value and positional embeddings with learnable weights."""

    def __init__(
            self,
            d_model: int,
            n_features: int,
            max_seq_len: int = 5000,
            dropout: float = 0.1
    ):
        super().__init__()

        self.value_embedding = ValueEmbedding(d_model, n_features)
        self.positional_embedding = PositionalEmbedding(
            d_model,
            max_seq_len,
            dropout
        )

        # Learnable weights for combining embeddings
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, features]
        Returns:
            Combined embedding tensor of shape [batch_size, seq_len, d_model]
        """
        value_embedded = self.value_embedding(x)
        pos_embedding = self.positional_embedding(value_embedded)

        return self.alpha * value_embedded + self.beta * pos_embedding