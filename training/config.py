# path: training/config.py
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    """Base training configuration."""
    learning_rate: float
    max_epochs: int
    use_early_stopping: bool
    early_stopping_patience: int
    batch_size: int
    device: str
    gradient_clip_val: float = 1.0

@dataclass
class TransformerTrainingConfig:
    """Training configuration specific to transformer models."""
    # Required fields (no defaults)
    learning_rate: float
    max_epochs: int
    use_early_stopping: bool
    early_stopping_patience: int
    batch_size: int
    device: str
    transformer_labels_count: int
    forecasting_horizon: int
    transformer_use_teacher_forcing: bool
    # Optional fields with defaults last
    gradient_clip_val: float = 1.0
