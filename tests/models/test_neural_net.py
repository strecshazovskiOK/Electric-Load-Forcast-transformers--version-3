import pytest
import torch
from models.architectures.neural_nets.simple_nn import SimpleNeuralNet

@pytest.fixture
def nn_config():
    return {
        'input_features': 10,
        'hidden_dims': [64, 32],
        'output_dim': 1,
        'dropout': 0.1,
        'activation': 'relu'
    }

def test_neural_net_initialization(nn_config):
    model = SimpleNeuralNet(nn_config)
    assert model.get_input_dims() == 10
    assert model.get_output_dims() == 1
    assert len(model.hidden_dims) == 2
    assert model.hidden_dims == [64, 32]

def test_neural_net_forward(nn_config):
    model = SimpleNeuralNet(nn_config)
    batch_size = 8
    x = torch.randn(batch_size, nn_config['input_features'])
    output = model(x)
    assert output.shape == (batch_size, nn_config['output_dim'])