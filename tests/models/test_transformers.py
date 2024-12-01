from __future__ import annotations  # Add this at the top
from typing import Dict, Any, Optional

# tests/models/test_transformers.py
import pytest
import torch
from models.architectures.transformers.vanilla_transformer import VanillaTransformer
from models.architectures.transformers.conv_transformer import ConvolutionalTransformer
from models.architectures.transformers.informer import Informer

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

@pytest.fixture
def transformer_config() -> Dict[str, Any]:
    return {
        'input_features': 10,
        'd_model': 64,
        'n_heads': 4,
        'n_encoder_layers': 2,
        'n_decoder_layers': 2,
        'd_ff': 256,
        'dropout': 0.1,
        'kernel_size': 3,
        'prob_sparse_factor': 5,
        'max_seq_len': 5000
    }

@pytest.mark.parametrize("transformer_class", [
    VanillaTransformer,
    ConvolutionalTransformer,
    Informer
])
def test_transformer_initialization(transformer_class: type, transformer_config: Dict[str, Any]) -> None:
    model = transformer_class(transformer_config)
    assert model.get_input_dims() == transformer_config['input_features']
    assert model.get_output_dims() == 1
    assert model.d_model == transformer_config['d_model']
    assert model.n_heads == transformer_config['n_heads']

@pytest.mark.parametrize("transformer_class", [
    VanillaTransformer,
    ConvolutionalTransformer,
    Informer
])
def test_transformer_forward(transformer_class: type, transformer_config: Dict[str, Any]) -> None:
    model = transformer_class(transformer_config)
    batch_size, seq_len = 4, 16  # Keep these values

    # Create input tensors (batch_first=False for PyTorch attention)
    src = torch.randn(seq_len, batch_size, transformer_config['input_features'])
    tgt = torch.randn(seq_len, batch_size, transformer_config['input_features'])

    # Create masks
    src_mask = None
    tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    tgt_mask = tgt_mask.to(src.device)

    # Create padding masks with correct shape (batch_size, seq_len)
    src_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(src.device)
    tgt_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(src.device)

    try:
        output = model(
            src,
            tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # Add assertions about output shape
        expected_shape = (seq_len, batch_size, 1)  # Since output_projection produces 1 feature
        assert output.shape == expected_shape, \
            f"Expected output shape {expected_shape}, got {output.shape}"
            
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
        print(f"Model class: {transformer_class.__name__}")
        raise