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
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.n_encoder_layers = config['n_encoder_layers']
        self.n_decoder_layers = config['n_decoder_layers']
        self.d_ff = config['d_ff']
        self.dropout = config.get('dropout', 0.1)
        self.input_features = config['input_features']
        self.output_dim = config.get('output_features', 1)

        # Initialize transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.n_heads,
            num_encoder_layers=self.n_encoder_layers,
            num_decoder_layers=self.n_decoder_layers,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True
        )

        # Initialize embeddings
        self.encoder_embedding = TotalEmbedding(
            d_model=self.d_model,
            value_features=1,  # Main time series value
            time_features=self.input_features - 1,  # Additional features
            dropout=self.dropout
        )
        self.decoder_embedding = TotalEmbedding(
            d_model=self.d_model,
            value_features=1,
            time_features=self.input_features - 1,
            dropout=self.dropout
        )

        # Output projection
        self.projection = nn.Linear(self.d_model, self.output_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, *args: Tensor, **kwargs: Any) -> Tensor:
        """
        Execute the model for the given input.
        
        Handles both:
        1. (x_enc, x_dec) format for explicit encoder-decoder inputs
        2. (x,) format for standard pipeline compatibility
        
        Args:
            *args: Either (x_enc, x_dec) or (x,)
            **kwargs: Additional arguments like masks
            
        Returns:
            Predictions tensor of shape [batch_size, seq_length, output_dim]
        """
        # Handle different input formats
        if len(args) == 2:
            x_enc, x_dec = args
            src_mask = kwargs.get('src_mask', None)
            tgt_mask = kwargs.get('tgt_mask', None)
        elif len(args) == 1:
            x = args[0]
            # Split input sequence for encoder and decoder
            split_point = x.size(1) - self.config.get('transformer_labels_count', 48)
            x_enc = x[:, :split_point, :]
            x_dec = x[:, split_point:, :]
            # Create appropriate masks
            src_mask = None
            tgt_mask = self.create_subsequent_mask(x_dec.size(1)).to(x.device)
        else:
            raise ValueError("Invalid number of arguments")

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

    def get_input_dims(self) -> int:
        """Return the number of input features."""
        return self.input_features

    def get_output_dims(self) -> int:
        """Return the number of output features."""
        return self.output_dim

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
            for _ in range(forecast_length):
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