# tests/unit/test_training_base.py
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR

from training.base import BaseTrainer, TrainingReport
from training.config import TrainingConfig

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

class MockTrainer(BaseTrainer):
    """Mock trainer for testing BaseTrainer functionality."""
    def train_phase(self, device: str) -> float:
        return 0.5

    def validation_phase(self, device: str) -> float:
        return 0.3


@pytest.fixture
def training_config():
    return TrainingConfig(
        learning_rate=0.001,
        max_epochs=10,
        use_early_stopping=True,
        early_stopping_patience=3,
        batch_size=32,
        device='cpu'
    )

@pytest.fixture
def mock_data():
    x_train = torch.randn(100, 10)
    y_train = torch.randn(100, 1)
    x_val = torch.randn(20, 10)
    y_val = torch.randn(20, 1)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)

    return train_loader, val_loader

@pytest.fixture
def mock_trainer(mock_data, training_config):
    train_loader, val_loader = mock_data
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=training_config.learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    return MockTrainer(
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        training_config.max_epochs,
        scheduler,
        training_config
    )

def test_base_trainer_initialization(mock_trainer):
    assert isinstance(mock_trainer.model, nn.Module)
    assert isinstance(mock_trainer.loss_criterion, nn.Module)
    assert isinstance(mock_trainer.optimizer, Optimizer)
    assert isinstance(mock_trainer.learning_rate_scheduler, torch.optim.lr_scheduler.StepLR)
    assert mock_trainer.epochs_count == 10

def test_training_report_serialization():
    report = TrainingReport(
        train_losses=[1.0, 0.8],
        val_losses=[0.9, 0.7],
        epochs=2
    )
    serialized = report.serialize()

    assert 'train_losses' in serialized
    assert len(serialized['train_losses']) == 2
    assert 'val_losses' in serialized
    assert len(serialized['val_losses']) == 2
    assert serialized['epochs'] == 2

def test_early_stopping(mock_trainer):
    report = mock_trainer.train()
    assert isinstance(report, TrainingReport)
    # Check train losses exist and their count is less than or equal to max epochs
    assert len(report.train_losses) <= mock_trainer.epochs_count
    # Validate that early stopping data is present
    assert report.early_stopping_epoch is not None

