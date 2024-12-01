# pipeline/config/pipeline_config.py
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Tuple
from datetime import date
from data_loading.base.base_dataset import DatasetConfig
from models.config.model_config import ModelConfig
from training.config import TrainingConfig

@dataclass
class PipelineConfig:
    """Pipeline configuration combining dataset, model, and training settings."""

    # Core configurations
    dataset_config: DatasetConfig
    model_config: ModelConfig
    training_config: TrainingConfig

    # Data paths and settings
    data_path: Path
    model_save_path: Optional[Path] = None
    experiment_save_path: Path = Path("experiments")

    # Data split dates
    train_dates: Tuple[date, date] = (
        date(2015, 10, 29),  # Start date of your data
        date(2016, 10, 28)   # ~70% of data
    )
    val_dates: Tuple[date, date] = (
        date(2016, 10, 29),  # ~15% of data
        date(2017, 1, 28)
    )
    test_dates: Tuple[date, date] = (
        date(2017, 1, 29),   # ~15% of data
        date(2017, 3, 12)    # End date of your data
    )

    # Add data format configuration
    time_resolution_minutes: int = 15  # New field to specify data granularity
    
    def get(self, key: str, default: Any = None) -> Any:
        """Helper method to safely get configuration values."""
        config_dict = asdict(self)
        # Check in all nested configs
        for config in [self.dataset_config, self.model_config, self.training_config]:
            if hasattr(config, key):
                return getattr(config, key)
        return config_dict.get(key, default)