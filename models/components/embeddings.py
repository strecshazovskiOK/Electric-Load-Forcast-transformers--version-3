import math
import torch
from torch import nn, Tensor
from typing import Optional
from utils.logging.logger import Logger

# Get module logger
logger = Logger.get_logger(__name__)

import math
import torch
from torch import nn, Tensor

class ValueEmbedding(nn.Module):
    def __init__(self, d_model: int, input_features: int):
        super().__init__()
        self.d_model = d_model
        self.input_features = input_features
        self.debug_counter = 0
        logger.debug(f"Initializing ValueEmbedding: d_model={d_model}, features={input_features}")
        
        self.linear = nn.Linear(input_features, d_model)
        
        with torch.no_grad():
            nn.init.xavier_uniform_(self.linear.weight, gain=1/math.sqrt(input_features))
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        # Print debug info only every 100 calls
        self.debug_counter += 1
        if self.debug_counter % 100 == 1:
            logger.debug(
                f"ValueEmbedding forward - "
                f"shape: {x.shape}, "
                f"range: [{x.min().item():.2f}, {x.max().item():.2f}], "
                f"device: {x.device}"
            )
        
        if x.size(-1) != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} input features but got {x.size(-1)}"
            )
            
        # Move linear layer to input device if needed
        if x.device != self.linear.weight.device:
            logger.warning(
                f"Device mismatch - Input: {x.device}, Layer: {self.linear.weight.device}. "
                "Moving layer to match input device."
            )
            self.linear = self.linear.to(x.device)
            
        return self.linear(x)
    


class CombinedEmbedding(nn.Module):
    def __init__(self, d_model: int, input_features: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.debug_counter = 0
        logger.debug(
            f"Initializing CombinedEmbedding - "
            f"d_model: {d_model}, features: {input_features}, "
            f"seq_len: {max_seq_len}, dropout: {dropout}"
        )
        
        self.d_model = d_model
        self.value_embedding = ValueEmbedding(d_model, input_features)
        
        # Create positional encoding
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        self.debug_counter += 1
        debug_enabled = self.debug_counter % 100 == 1

        try:
            if debug_enabled:
                logger.debug(
                    f"CombinedEmbedding forward - "
                    f"input shape: {x.shape}, "
                    f"device: {x.device}"
                )
            
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
                
            # Get value embeddings (this will handle device consistency internally)
            value_emb = self.value_embedding(x)
            
            # Ensure positional encoding is on the same device
            seq_len = x.size(1)
            pos_emb = self.pe[:, :seq_len].to(x.device)
            
            # Combine embeddings
            combined = value_emb + pos_emb
            
            return self.dropout(combined)
            
        except Exception as e:
            logger.error(f"CombinedEmbedding forward failed: {str(e)}")
            logger.debug("Device mapping:")
            logger.debug(f"Input: {x.device}")
            logger.debug(f"Value embedding: {next(self.value_embedding.parameters()).device}")
            logger.debug(f"PE buffer: {self.pe.device}")
            raise
            
    def _create_positional_encoding(self, max_len: int, d_model: int) -> Tensor:
        """Create enhanced positional encoding with proper initialization."""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return pe

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
