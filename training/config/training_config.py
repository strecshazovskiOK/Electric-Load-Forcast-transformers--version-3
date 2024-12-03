from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import warnings

@dataclass
class BaseConfig:
    """Base configuration with required fields."""
    learning_rate: float
    max_epochs: int
    use_early_stopping: bool
    early_stopping_patience: int

@dataclass 
class ResolutionBasedTrainingConfig(BaseConfig):
    """Training configuration that adapts to different time resolutions."""
    # Time resolution parameters
    input_resolution_minutes: int
    forecast_resolution_minutes: int
    
    # Optional fields with defaults after
    batch_size: int = 32
    device: str = 'cuda'
    gradient_clip_val: float = 1.0
    
    def __post_init__(self):
        if self.forecast_resolution_minutes <= 60:
            self.batch_size = max(64, self.batch_size)
            if self.use_early_stopping:
                self.early_stopping_patience = max(15, self.early_stopping_patience)
        elif self.forecast_resolution_minutes >= 1440:
            self.batch_size = min(16, self.batch_size)
            if self.use_early_stopping:
                self.early_stopping_patience = min(8, self.early_stopping_patience)

@dataclass
class TransformerTrainingConfig(ResolutionBasedTrainingConfig):
    """Training configuration specific to transformer models."""
    # Transform-specific fields with explicit defaults
    transformer_labels_count: int = field(
        default=24,
        metadata={"help": "Number of labels for transformer model"}
    )
    forecasting_horizon: int = field(
        default=24,
        metadata={"help": "Number of time steps to forecast"}
    )
    transformer_use_teacher_forcing: bool = field(
        default=True,
        metadata={"help": "Whether to use teacher forcing during training"}
    )
    
    # Model training parameters
    attention_dropout: float = 0.1
    optimizer: str = 'adamw'
    scheduler: Optional[str] = 'one_cycle'
    optimizer_config: Dict[str, Any] = field(default_factory=lambda: {
        'weight_decay': 0.01,
        'betas': (0.9, 0.98)
    })
    scheduler_config: Dict[str, Any] = field(default_factory=lambda: {
        'pct_start': 0.3,
        'div_factor': 25.0,
        'final_div_factor': 1000.0
    })

    def __post_init__(self):
        """Adjust transformer-specific parameters and validate configuration."""
        super().__post_init__()  # Call parent's post init first
        
        # Remove warnings since we're now explicitly setting these values
        # with proper metadata
        
        # Validate values
        if self.transformer_labels_count <= 0:
            raise ValueError("transformer_labels_count must be positive")
        if self.forecasting_horizon <= 0:
            raise ValueError("forecasting_horizon must be positive")
            
        # Adjust attention dropout based on resolution
        if self.forecast_resolution_minutes <= 60:
            self.attention_dropout = min(0.3, self.attention_dropout)
        elif self.forecast_resolution_minutes >= 1440:
            self.attention_dropout = max(0.1, self.attention_dropout)