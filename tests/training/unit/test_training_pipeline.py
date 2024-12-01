# tests/integration/test_training_pipeline.py
import pytest
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from training.callbacks import EarlyStopping, ModelCheckpoint
from training.callbacks.learning_rate_scheduler import LRSchedulerCallback
from training.config.training_config import TrainingConfig, TransformerTrainingConfig
from training.reports.training_report import TrainingReport
from training.trainers.neural_net_trainer import NeuralNetTrainer
from training.trainers.transformer_trainer import TransformerTrainer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def simple_dataset():
    # Create a simple dataset for testing
    x = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32)

@pytest.fixture
def neural_net_config():
    return TrainingConfig(
        learning_rate=0.001,
        max_epochs=5,
        use_early_stopping=True,
        early_stopping_patience=2,
        batch_size=32,
        device='cpu'
    )

def test_neural_net_training_pipeline(simple_dataset, neural_net_config):
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=neural_net_config.learning_rate)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    trainer = NeuralNetTrainer(
        simple_dataset,
        simple_dataset,
        model,
        criterion,
        optimizer,
        neural_net_config.max_epochs,
        scheduler,
        neural_net_config
    )

    report = trainer.train()
    assert isinstance(report, TrainingReport)
    assert hasattr(report, 'epochs') and report.epochs

def test_transformer_training_pipeline(simple_dataset):
    config = TransformerTrainingConfig(
        learning_rate=0.001,
        max_epochs=5,
        use_early_stopping=True,
        early_stopping_patience=2,
        batch_size=32,
        device='cpu',
        transformer_labels_count=1,
        forecasting_horizon=12
    )

    # Create proper transformer input tensors
    seq_length = config.transformer_labels_count + config.forecasting_horizon
    x = torch.randn(100, 10, 2)  # [batch_size, seq_len, features]
    y = torch.randn(100, seq_length, 2)  # [batch_size, seq_len, 2] (target value + features)
    dataset = TensorDataset(x, y)
    transformer_dataset = DataLoader(dataset, batch_size=32)

    class MockTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, src, tgt, src_mask=None, tgt_mask=None):
            return self.linear(tgt)

    model = MockTransformer()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    trainer = TransformerTrainer(
        transformer_dataset,
        transformer_dataset,
        model,
        criterion,
        optimizer,
        config.max_epochs,
        scheduler,
        config
    )

    report = trainer.train()
    assert isinstance(report, TrainingReport)
    assert hasattr(report, 'epochs') and report.epochs


def test_training_with_all_callbacks(simple_dataset, neural_net_config, tmp_path):
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=neural_net_config.learning_rate)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    # Create all callbacks
    early_stopping = EarlyStopping(patience=2)
    filepath = str(tmp_path / "checkpoint-{epoch:02d}-{val_loss:.2f}.pt")
    model_checkpoint = ModelCheckpoint(
        filepath=filepath,
        save_best_only=True,
        monitor='val_loss',
    )
    lr_scheduler = LRSchedulerCallback(scheduler)

    trainer = NeuralNetTrainer(
        simple_dataset,
        simple_dataset,
        model,
        criterion,
        optimizer,
        neural_net_config.max_epochs,
        scheduler,
        neural_net_config
    )

    # Set up callbacks properly
    early_stopping.model = model
    model_checkpoint.model = model
    trainer.callbacks = [early_stopping, model_checkpoint, lr_scheduler]
    
    # Train and ensure at least one epoch completes
    report = trainer.train()
    assert isinstance(report, TrainingReport)
    
    # Wait for file operations and verify checkpoints
    import time
    time.sleep(1)  # Increased wait time
    
    checkpoint_files = list(tmp_path.glob("checkpoint-*.pt"))
    assert len(checkpoint_files) > 0, f"No checkpoint files found in {tmp_path}"
    
    # Verify checkpoint file format
    checkpoint_file = checkpoint_files[0]
    assert checkpoint_file.exists(), f"Checkpoint file {checkpoint_file} does not exist"
    assert str(checkpoint_file).startswith(str(tmp_path)), "Checkpoint not in correct directory"