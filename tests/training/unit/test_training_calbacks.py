# tests/unit/test_training_callbacks.py
import pytest
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR

from training.callbacks.early_stopping import EarlyStopping
from training.callbacks.learning_rate_scheduler import LRSchedulerCallback
from training.callbacks.model_checkpoint import ModelCheckpoint



class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def sample_model():
    return SimpleModel()

@pytest.fixture
def sample_logs():
    return {'val_loss': 0.5, 'train_loss': 0.4}

def test_early_stopping_callback(sample_model):
    early_stopping = EarlyStopping(patience=2, monitor='val_loss')

    # Initialize with model (removed explicit model assignment)
    early_stopping.on_training_begin(sample_model, {})

    assert early_stopping.wait == 0
    assert early_stopping.best_value == float('inf')

    # Test improvement
    early_stopping.on_epoch_end(0, {'val_loss': 0.5})
    assert early_stopping.wait == 0
    assert early_stopping.best_value == 0.5

    # Test no improvement
    early_stopping.on_epoch_end(1, {'val_loss': 0.6})
    assert early_stopping.wait == 1

    # Test early stopping
    early_stopping.on_epoch_end(2, {'val_loss': 0.7})
    assert early_stopping.stopped_epoch == 2

def test_model_checkpoint_callback(tmp_path, sample_model, sample_logs):
    filepath = str(tmp_path / "checkpoint-{epoch:02d}-{val_loss:.2f}.pt")
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        save_best_only=True
    )

    # Initialize with model
    checkpoint.on_training_begin(sample_model, {})
    checkpoint.on_epoch_end(0, sample_logs)

    # Check if file exists using correct format
    expected_file = tmp_path / f"checkpoint-00-{sample_logs['val_loss']:.2f}.pt"
    assert expected_file.exists()

def test_lr_scheduler_callback(sample_model):
    optimizer = Adam(sample_model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    callback = LRSchedulerCallback(scheduler)
    callback = LRSchedulerCallback(scheduler)

    initial_lr = optimizer.param_groups[0]['lr']
    callback.on_epoch_end(0, {'val_loss': 0.5})
    new_lr = optimizer.param_groups[0]['lr']

    assert new_lr == initial_lr * 0.1

