# tests/models/test_base.py
import pytest
import torch

from models.base.base_model import BaseModel

class MockModel(BaseModel):
    """Mock model for testing BaseModel functionality."""
    def __init__(self, config):
        super().__init__(config)
        self.input_dim = config['input_features']
        self.output_dim = config['output_dim']
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        return self.linear(x)

    def get_input_dims(self):
        return self.input_dim

    def get_output_dims(self):
        return self.output_dim

def test_base_model_initialization():
    config = {'input_features': 10, 'output_dim': 1}
    model = MockModel(config)
    assert model.get_input_dims() == 10
    assert model.get_output_dims() == 1
    assert model.get_model_config() == config