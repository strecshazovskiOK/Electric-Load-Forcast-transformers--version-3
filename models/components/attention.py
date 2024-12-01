# models/components/attention.py
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            dropout: float = 0.1,
            bias: bool = True
    ):
        # Important: Call parent's init first!
        nn.Module.__init__(self)

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        # Handle sequence first format (seq_len, batch_size, features)
        query = query.transpose(0, 1)  # Convert to (batch_size, seq_len, features)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        
        batch_size = query.size(0)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                attn_mask = attn_mask.expand(batch_size, self.n_heads, attn_mask.size(-2), attn_mask.size(-1))
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        if key_padding_mask is not None:
            # Reshape padding mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # Expand to [batch_size, n_heads, tgt_len, src_len]
            key_padding_mask = key_padding_mask.expand(-1, self.n_heads, scores.size(2), -1)
            scores = scores.masked_fill(key_padding_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(context)
        
        # Convert back to sequence first format
        output = output.transpose(0, 1)

        return output, attn if need_weights else None

class ConvolutionalAttention(nn.Module):
    """Attention mechanism with convolutional processing."""

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            kernel_size: int = 3,
            dropout: float = 0.1
    ):
        # Important: Call parent's init first!
        nn.Module.__init__(self)

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        padding = (kernel_size - 1) // 2

        self.query_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=padding,
            padding_mode='replicate'
        )
        self.key_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=padding,
            padding_mode='replicate'
        )
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        # Apply convolution
        query = self.query_conv(query.transpose(-1, -2)).transpose(-1, -2)
        key = self.key_conv(key.transpose(-1, -2)).transpose(-1, -2)

        # Use attention with same signature
        return self.attention(query, key, value,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=need_weights)



class ProbSparseAttention(nn.Module):
    """Probabilistic sparse attention mechanism."""

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            factor: int = 5,
            dropout: float = 0.1
    ):
        # Important: Call parent's init first!
        nn.Module.__init__(self)

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.factor = factor

    def _prob_QK(self, Q: Tensor, K: Tensor, sample_k: int, n_top: int) -> Tuple[Tensor, Tensor]:
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # Calculate Q_K
        Q_K = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)

        # Find top k queries
        M = Q_K.max(-1)[0] - torch.div(Q_K.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Reduce Q to top k queries
        Q_reduce = torch.gather(Q, 2, M_top.unsqueeze(-1).expand(-1, -1, -1, D))

        return Q_reduce, M_top

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        return self.attention(query, key, value,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=need_weights)

