# tests/unit/test_callbacks.py
from typing import Union
import pytest
from pathlib import Path
import torch.optim as optim
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LRSchedulerCallback
)
from models.architectures.linear.linear_regression import LinearRegression


@pytest.fixture
def real_model():
    """Create a real LinearRegression model instance."""
    from typing import Dict, Union

    model_config: Dict[str, Union[int, bool]] = {
        'input_features': 10,
        'output_dim': 1,
        'zero_init_bias': True
    }
    return LinearRegression(model_config)

@pytest.fixture

def optimizer(real_model: LinearRegression):
    return Adam(real_model.parameters(), lr=0.001)

@pytest.fixture
def scheduler(optimizer: Optimizer):
    """Create learning rate scheduler."""
    return optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

def test_early_stopping_initialization():
    """Test EarlyStopping callback initialization."""
    early_stopping = EarlyStopping(patience=3, monitor='val_loss', min_delta=0.01)
    assert early_stopping.patience == 3
    assert early_stopping.monitor == 'val_loss'
    assert early_stopping.min_delta == 0.01
    assert early_stopping.wait == 0
    assert early_stopping.best_value == float('inf')
    assert early_stopping.stopped_epoch == 0

def test_early_stopping_improving_loss(real_model: LinearRegression):
    """Test EarlyStopping behavior with improving loss."""
    early_stopping = EarlyStopping(patience=3, monitor='val_loss')

    # Initialize callback with model
    config = {'model': real_model}
    early_stopping.on_training_begin(real_model, config)

    # Test improving loss
    logs = {'val_loss': 0.5}
    early_stopping.on_epoch_end(1, logs)
    assert early_stopping.wait == 0
    assert early_stopping.best_value == 0.5

    # Further improvement
    logs = {'val_loss': 0.3}
    early_stopping.on_epoch_end(2, logs)
    assert early_stopping.wait == 0
    assert early_stopping.best_value == 0.3

def test_early_stopping_patience(real_model: LinearRegression):
    """Test EarlyStopping patience behavior."""
    early_stopping = EarlyStopping(patience=2, monitor='val_loss')
    config = {'model': real_model}
    early_stopping.on_training_begin(real_model, config)

    # Initial loss
    logs = {'val_loss': 0.5}
    early_stopping.on_epoch_end(1, logs)

    # Worse losses - should increment wait counter
    for i in range(2, 5):
        logs = {'val_loss': 0.5 + i*0.1}
        early_stopping.on_epoch_end(i, logs)

    assert early_stopping.wait >= early_stopping.patience
    assert early_stopping.stopped_epoch > 0
def test_model_checkpoint_initialization(tmp_path: Path):
    """Test ModelCheckpoint initialization."""
    filepath = str(tmp_path / "checkpoint-{epoch:02d}-{loss:.2f}.pt")
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True
    )

    assert checkpoint.monitor == 'val_loss'
    assert checkpoint.save_best_only
    assert checkpoint.save_weights_only
    assert checkpoint.best_value == float('inf')

def test_model_checkpoint_saving(real_model: LinearRegression, tmp_path: Path):
    """Test ModelCheckpoint saving functionality."""
    filepath = str(Path(tmp_path) / "checkpoint-{epoch}-{loss:.2f}.pt")
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        save_best_only=True
    )

    # Initialize callback with model
    config = {'model': real_model}
    checkpoint.on_training_begin(real_model, config)

    # Test saving with improving loss
    logs = {'val_loss': 0.5, 'loss': 0.5}  # Added loss to match filepath format
    checkpoint.on_epoch_end(1, logs)

    expected_file = tmp_path / "checkpoint-1-0.50.pt"
    assert expected_file.exists()

def test_lr_scheduler_callback(real_model: LinearRegression, optimizer: Optimizer, scheduler: optim.lr_scheduler.LRScheduler):
    """Test LRSchedulerCallback functionality."""
    callback = LRSchedulerCallback(scheduler, monitor='val_loss')

    # Test initialization
    assert callback.scheduler == scheduler
    assert callback.monitor == 'val_loss'

    # Test LR scheduling
    logs = {'val_loss': 0.5}

    # Simulate epoch end
    callback.on_epoch_end(1, logs)

    new_lr = optimizer.param_groups[0]['lr']
    assert new_lr == scheduler.get_last_lr()[0]

def test_lr_scheduler_plateau(real_model: LinearRegression):
    """Test LRSchedulerCallback with ReduceLROnPlateau."""
    optimizer = Adam(real_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=1
    )
    callback = LRSchedulerCallback(scheduler, monitor='val_loss')

    # Initial learning rate
    initial_lr = optimizer.param_groups[0]['lr']

    # Simulate several epochs with non-improving loss
    for i in range(3):
        logs = {'val_loss': 0.5 + i*0.1}
        callback.on_epoch_end(i, logs)

    # Check that LR has been reduced
    assert optimizer.param_groups[0]['lr'] < initial_lr

def test_callbacks_integration(tmp_path: Path, real_model: LinearRegression, optimizer: Optimizer, scheduler: optim.lr_scheduler.LRScheduler):
    """Test integration of multiple callbacks."""
    # Initialize callbacks
    early_stopping = EarlyStopping(patience=2, monitor='val_loss')
    checkpoint = ModelCheckpoint(
        filepath=str(tmp_path / "checkpoint-{epoch}-{loss:.2f}.pt"),
        monitor='val_loss',
        save_best_only=True
    )
    lr_scheduler = LRSchedulerCallback(scheduler, monitor='val_loss')

    callbacks: list[Union[EarlyStopping, ModelCheckpoint, LRSchedulerCallback]] = [early_stopping, checkpoint, lr_scheduler]

    # Initialize all callbacks with model in config
    config = {'model': real_model}
    for callback in callbacks:
        callback.on_training_begin(real_model, config)

    # Simulate training epochs
    logs = {'val_loss': 0.5, 'loss': 0.5}  # Added loss to match filepath format
    for callback in callbacks:
        callback.on_epoch_end(1, logs)

    # Verify checkpoint was saved
    expected_file = tmp_path / "checkpoint-1-0.50.pt"
    assert expected_file.exists()

    # Verify learning rate was updated
    assert optimizer.param_groups[0]['lr'] == scheduler.get_last_lr()[0]