# training/config/training_config.py
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class TrainingConfig:
    """Base training configuration."""
    learning_rate: float
    max_epochs: int
    use_early_stopping: bool
    early_stopping_patience: int
    batch_size: int = 32
    device: str = 'cuda'

@dataclass
class NeuralNetTrainingConfig(TrainingConfig):
    """Neural network specific training configuration."""
    learning_rate_scheduler_step: int = 30
    learning_rate_scheduler_gamma: float = 0.1
    gradient_clipping: Optional[float] = None

@dataclass
class TransformerTrainingConfig(TrainingConfig):
    """Transformer specific training configuration."""
    transformer_labels_count: int = 1
    forecasting_horizon: int = 24
    transformer_use_teacher_forcing: bool = False
    transformer_use_auto_regression: bool = False
    learning_rate_scheduler_step: int = 30
    learning_rate_scheduler_gamma: float = 0.1
    attention_dropout: float = 0.1
    gradient_clipping: Optional[float] = None  # Added this line
    optimizer: str = 'adam'
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    scheduler: Optional[str] = None
    scheduler_config: Dict[str, Any] = field(default_factory=dict)

    