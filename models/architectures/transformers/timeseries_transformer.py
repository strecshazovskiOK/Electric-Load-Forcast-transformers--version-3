from typing import Dict, Any, List, Optional, Tuple, Union
import torch
from torch import nn, Tensor

from models.registry.factory import ModelFactory
from models.registry.model_types import ModelType
from models.components.embeddings import TotalEmbedding
from models.base.base_model import BaseModel  # Add this import


def average_attention_scores(attention_scores: List[torch.Tensor]) -> torch.Tensor:
    """Calculate average attention scores across layers."""
    return torch.mean(torch.stack(attention_scores), dim=0)


@ModelFactory.register(ModelType.TIME_SERIES_TRANSFORMER)
class TimeSeriesTransformer(BaseModel):  # Change parent class to BaseModel
    """Transformer model specifically designed for time series forecasting."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)  # Call BaseModel's __init__
        
        # Extract configuration
        d_model = config['d_model']
        n_heads = config['n_heads']
        n_encoder_layers = config['n_encoder_layers']
        n_decoder_layers = config['n_decoder_layers']
        d_ff = config['d_ff']
        dropout = config['dropout']
        input_features = config['input_features']

        # Initialize transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True  # Important for time series data
        )

        # Initialize embeddings
        self.encoder_embedding = TotalEmbedding(
            d_model=d_model,
            value_features=1,  # Main time series value
            time_features=input_features - 1,  # Additional features
            dropout=dropout
        )
        self.decoder_embedding = TotalEmbedding(
            d_model=d_model,
            value_features=1,
            time_features=input_features - 1,
            dropout=dropout
        )

        # Output projection
        self.projection = nn.Linear(d_model, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(
        self,
        x_enc: Tensor,
        x_dec: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Execute the model for the given input.

        Args:
            x_enc: Raw input for encoder [batch_size, seq_enc_length, features]
            x_dec: Raw input for decoder [batch_size, seq_dec_length, features]
            src_mask: Optional mask for encoder
            tgt_mask: Optional mask for decoder (usually needed for autoregressive prediction)

        Returns:
            Predictions tensor of shape [batch_size, seq_dec_length, 1]
        """
        # Apply embeddings
        enc_embedding = self.encoder_embedding(x_enc)
        dec_embedding = self.decoder_embedding(x_dec)

        # Apply transformer
        out = self.transformer(
            enc_embedding,
            dec_embedding,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )

        # Project to output dimension
        out = self.projection(self.relu(out))
        return out

    def get_cross_attention_scores(self) -> torch.Tensor:
        """Get average cross-attention scores across all decoder layers."""
        return average_attention_scores([
            layer.multihead_attn.attention_weights
            for layer in self.transformer.decoder.layers
        ])

    def get_self_attention_scores(self) -> torch.Tensor:
        """Get average self-attention scores across all decoder layers."""
        return average_attention_scores([
            layer.self_attn.attention_weights
            for layer in self.transformer.decoder.layers
        ])

    def create_subsequent_mask(self, size: int) -> Tensor:
        """Create causal mask to prevent attending to future tokens."""
        return self.transformer.generate_square_subsequent_mask(size)

    def predict_sequence(
        self,
        src: Tensor,
        forecast_length: int,
        return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Generate multi-step predictions.

        Args:
            src: Source sequence [batch_size, seq_length, features]
            forecast_length: Number of steps to predict
            return_attention: Whether to return attention weights

        Returns:
            predictions: Predicted sequence
            attention_weights: Optional attention weights if return_attention=True
        """
        self.eval()
        device = src.device

        with torch.no_grad():
            # Initial embedding
            enc_out = self.encoder_embedding(src)

            # Start with the last value of source sequence
            dec_input = src[:, -1:, :]
            predictions = []
            attention_weights = []

            # Generate predictions autoregressively
            for i in range(forecast_length):
                # Create appropriate masking
                tgt_mask = self.create_subsequent_mask(dec_input.size(1)).to(device)

                # Get decoder embedding
                dec_emb = self.decoder_embedding(dec_input)

                # Get transformer output
                out = self.transformer(
                    enc_out,
                    dec_emb,
                    tgt_mask=tgt_mask
                )

                # Project and store prediction
                pred = self.projection(self.relu(out[:, -1:, :]))
                predictions.append(pred)

                if return_attention:
                    attention_weights.append(self.get_cross_attention_scores())

                # Update decoder input for next step
                dec_input = torch.cat([dec_input, pred], dim=1)

            predictions = torch.cat(predictions, dim=1)

            if return_attention:
                return predictions, torch.stack(attention_weights)
            return predictions