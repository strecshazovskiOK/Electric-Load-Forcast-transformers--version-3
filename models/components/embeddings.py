import math
import torch
from torch import nn, Tensor
from typing import Optional

class ValueEmbedding(nn.Module):
    """Projects time series values to model dimension."""

    def __init__(self, d_model: int, time_series_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(time_series_features, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Creates from the given tensor a linear projection.

        Args:
            x: the input tensor to project, shape: [batch_size, sequence_length, features]
        Returns:
            the projected tensor of shape: [batch_size, sequence_length, model_dimension]
        """
        return self.linear(x)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe.requires_grad_(False)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Creates a positional encoding for the given tensor.

        Args:
            x: tensor for which the pe is created, shape: [batch_size, sequence_length, model_dimension]
        Returns:
            positional encoding of dimension [1, sequence_length, model_dimension]
        """
        return self.pe[:, :x.size(1), :]


class TotalEmbedding(nn.Module):
    """Combines value and positional embeddings with learnable weights."""

    def __init__(self, d_model: int, value_features: int, time_features: int, dropout: float):
        super().__init__()

        self.value_embedding = ValueEmbedding(d_model, value_features + time_features)
        self.positional_encoding = PositionalEncoding(d_model)

        # Initialize learnable weights with ones for stable training
        self.linear_embedding_weight = nn.Linear(2, 1, bias=False)
        self.linear_embedding_weight.weight.data.fill_(1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Projects the given tensor x on the model_dimension (in the last dimension) and combines this with a positional
        encoding (PE). The PE is added with learned weights to the projected x tensor. Dropout is applied on the final
        result.

        Args:
            x: tensor of dimension [Batch_Size, Sequence_Length, Features]
        Returns:
            the embedded value of shape: [Batch_Size, Sequence_Length, model_dimension]
        """
        value_embedded = self.value_embedding(x)
        pe = self.positional_encoding(x).repeat(x.shape[0], 1, 1)

        # Combine embeddings with learned weights
        return self.dropout(
            self.linear_embedding_weight.weight[0][0] * value_embedded + 
            self.linear_embedding_weight.weight[0][1] * pe
        )


class CombinedEmbedding(nn.Module):
    """Alternative combined embedding with separate value and positional components."""

    def __init__(
            self,
            d_model: int,
            n_features: int,
            max_seq_len: int = 5000,
            dropout: float = 0.1
    ):
        super().__init__()

        # Core embeddings
        self.value_embedding = ValueEmbedding(d_model, n_features)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)

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
        pos_embedded = self.positional_encoding(x)

        return self.dropout(self.alpha * value_embedded + self.beta * pos_embedded)