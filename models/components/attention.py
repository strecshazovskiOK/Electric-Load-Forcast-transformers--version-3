# models/components/attention.py
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from utils.logging.logger import Logger
from utils.logging.config import LoggerConfig, LogLevel


class MultiHeadAttention(nn.Module):
    r"""Allows the model to jointly attend to information from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        # Initialize logger
        logger_config = LoggerConfig(
            level=LogLevel.INFO,
            component_name="MultiHeadAttention",
            include_timestamp=True
        )
        self.logger = Logger.get_logger(__name__, logger_config)
        
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self._reset_parameters()
        self.debug_counter = 0
        self.max_debug_prints = 3  # Only print first 3 times

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask=None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        # Only log debug info for first few calls
        if self.debug_counter < self.max_debug_prints:
            self.logger.debug(f"MultiHeadAttention forward pass (call {self.debug_counter + 1}/{self.max_debug_prints})", {
                "query_device": str(query.device),
                "key_device": str(key.device),
                "value_device": str(value.device),
                "attention_mask_device": str(attn_mask.device) if attn_mask is not None else "None",
                "key_padding_mask_device": str(key_padding_mask.device) if key_padding_mask is not None else "None"
            })
            self.debug_counter += 1

        # Ensure attn_mask is on the correct device
        if attn_mask is not None:
            attn_mask = attn_mask.to(query.device)


        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        return attn_output, attn_output_weights

class ConvolutionalAttention(nn.Module):
    """Attention mechanism with convolutional processing."""

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            kernel_size: int = 3,
            dropout: float = 0.1
    ):
        super().__init__()
        # Initialize logger
        logger_config = LoggerConfig(
            level=LogLevel.INFO,
            component_name="ConvolutionalAttention",
            include_timestamp=True
        )
        self.logger = Logger.get_logger(__name__, logger_config)
        
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

        # Use attention
        return self.attention(query, key, value,
                            key_padding_mask=key_padding_mask,
                            need_weights=need_weights,
                            attn_mask=attn_mask)

class ProbSparseAttention(nn.Module):
    """Probabilistic sparse attention mechanism."""

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            factor: int = 5,
            dropout: float = 0.1
    ):
        super().__init__()
        # Initialize logger
        logger_config = LoggerConfig(
            level=LogLevel.INFO,
            component_name="ProbSparseAttention",
            include_timestamp=True
        )
        self.logger = Logger.get_logger(__name__, logger_config)
        
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
                            key_padding_mask=key_padding_mask,
                            need_weights=need_weights,
                            attn_mask=attn_mask)
