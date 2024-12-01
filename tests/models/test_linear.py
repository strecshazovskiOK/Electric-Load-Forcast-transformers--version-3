# tests/models/test_linear.py
import pytest
import torch
from models.architectures.linear.linear_regression import LinearRegression

@pytest.fixture
def linear_config():
    return {
        'input_features': 5,
        'output_dim': 1,
        'zero_init_bias': True
    }

def test_linear_regression_initialization(linear_config):
    model = LinearRegression(linear_config)
    assert model.get_input_dims() == 5
    assert model.get_output_dims() == 1
    assert torch.allclose(model.linear.bias, torch.zeros_like(model.linear.bias))

def test_linear_regression_forward(linear_config):
    model = LinearRegression(linear_config)
    batch_size = 10
    x = torch.randn(batch_size, linear_config['input_features'])
    output = model(x)
    assert output.shape == (batch_size, linear_config['output_dim'])

