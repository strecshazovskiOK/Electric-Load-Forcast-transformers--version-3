# models/architectures/transformers/conv_transformer.py
from torch import nn
from typing import Dict, Any

from models.registry.factory import ModelFactory
from models.registry.model_types import ModelType

from .base_transformer import BaseTransformer
from ...components.layers import EncoderLayer, DecoderLayer


@ModelFactory.register(ModelType.CONV_TRANSFORMER)
class ConvolutionalTransformer(BaseTransformer):
    """Transformer with convolutional attention mechanism."""

    def __init__(self, config: Dict[str, Any]):
        self.kernel_size = config.get('kernel_size', 3)
        super().__init__(config)

    def _create_encoder_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            EncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                attention_type="convolutional",
                kernel_size=self.kernel_size,
                activation="relu"
            ) for _ in range(self.n_encoder_layers)
        ])

    def _create_decoder_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            DecoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                attention_type="convolutional",
                kernel_size=self.kernel_size,
                activation="relu"
            ) for _ in range(self.n_decoder_layers)
        ])