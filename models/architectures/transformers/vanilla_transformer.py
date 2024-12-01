import torch
from torch import nn
from typing import Dict, Any

from models.registry.factory import ModelFactory
from models.registry.model_types import ModelType
from models.architectures.transformers.base_transformer import BaseTransformer
from models.components.layers import EncoderLayer, DecoderLayer

@ModelFactory.register(ModelType.VANILLA_TRANSFORMER)
class VanillaTransformer(BaseTransformer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.input_dim = config['input_features']
        self.output_dim = config['input_features']  # Match input features for reconstruction
        self.seq_length = config.get('transformer_labels_count', 12)
        
        self.input_embedding = nn.Linear(self.input_dim, self.d_model)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, config.get('max_seq_length', 1000), self.d_model)
        )
        
        # Simplified output projection (per timestep)
        self.output_projection = nn.Linear(self.d_model, self.output_dim)

    def _create_encoder_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            EncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                attention_type="standard",
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
                attention_type="standard",
                activation="relu"
            ) for _ in range(self.n_decoder_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape

        # Embed input
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_length, :]
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        
        # Take only the last seq_length time steps
        x = x[:, -self.seq_length:, :]
        
        # Project to output dimension
        x = self.output_projection(x)
        
        return x