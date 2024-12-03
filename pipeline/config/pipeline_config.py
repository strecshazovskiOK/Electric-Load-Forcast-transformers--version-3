from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple
from datetime import date, timedelta

from data_loading.base.base_dataset import DatasetConfig
from models.config.model_config import ModelConfig
from models.registry.model_types import ModelType
from training.config.training_config import (
    BaseConfig,
    TransformerTrainingConfig,
    ResolutionBasedTrainingConfig
)

@dataclass
class PipelineConfig:
    """Pipeline configuration combining dataset, model, and training settings."""
    # Core configurations
    dataset_config: DatasetConfig
    model_config: ModelConfig
    training_config: BaseConfig  # Changed from TrainingConfig to BaseConfig

    # Data paths and settings
    data_path: Path
    model_save_path: Optional[Path] = None
    experiment_save_path: Path = Path("experiments")

    # Data split dates - updated to include validation dates
    train_dates: Tuple[date, date] = field(default_factory=lambda: (
        date(2015, 10, 29),
        date(2016, 10, 28)  # Adjusted to leave room for validation
    ))
    val_dates: Tuple[date, date] = field(default_factory=lambda: (
        date(2016, 10, 29),
        date(2017, 1, 28)
    ))
    test_dates: Tuple[date, date] = field(default_factory=lambda: (
        date(2017, 1, 29),
        date(2017, 3, 12)
    ))

    # Resolution configuration
    input_resolution_minutes: int = field(default=15)
    forecast_resolution_minutes: int = field(default=15)
    

    def __post_init__(self):
        """Validate and adjust configuration based on resolutions."""
        self._validate_resolutions()
        self._adjust_split_dates()
        self._synchronize_configs()

    def _validate_resolutions(self) -> None:
        """Validate resolution settings."""
        if self.input_resolution_minutes <= 0:
            raise ValueError("Input resolution must be positive")
        if self.forecast_resolution_minutes <= 0:
            raise ValueError("Forecast resolution must be positive")
        if self.forecast_resolution_minutes < self.input_resolution_minutes:
            raise ValueError("Forecast resolution cannot be finer than input resolution")

    def _adjust_split_dates(self) -> None:
        """Adjust split dates based on resolution."""
        # For monthly predictions, ensure splits align with month boundaries
        if self.forecast_resolution_minutes >= 43200:  # Monthly
            self.train_dates = (
                self.train_dates[0].replace(day=1),
                self.train_dates[1].replace(day=1) + self._get_month_delta()
            )
            self.val_dates = (
                self.val_dates[0].replace(day=1),
                self.val_dates[1].replace(day=1) + self._get_month_delta()
            )
            self.test_dates = (
                self.test_dates[0].replace(day=1),
                self.test_dates[1].replace(day=1) + self._get_month_delta()
            )

    def _synchronize_configs(self) -> None:
        """Ensure resolution settings are synchronized across all configs."""
        # Store original feature dimensions
        original_features = self.model_config.input_features
        
        # Update dataset config
        if isinstance(self.dataset_config, DatasetConfig):
            self.dataset_config.input_resolution_minutes = self.input_resolution_minutes
            self.dataset_config.forecast_resolution_minutes = self.forecast_resolution_minutes

        # Update model config with preserved features
        if self.model_config.model_type.is_resolution_specific:
            self.model_config = self._get_resolution_specific_model_config()
            # Ensure features are preserved
            self.model_config.input_features = original_features
            self.model_config.time_features = original_features - 1
            self.model_config.value_features = 1

        # Update training config
        if isinstance(self.training_config, TransformerTrainingConfig):
            self.training_config.input_resolution_minutes = self.input_resolution_minutes
            self.training_config.forecast_resolution_minutes = self.forecast_resolution_minutes

    def _get_month_delta(self) -> timedelta:
        """Get a month delta for date calculations."""
        return timedelta(days=30)  # Approximation

    def _get_resolution_specific_model_config(self) -> ModelConfig:
        
        """Get model configuration optimized for the current resolution."""
        model_type = ModelType.get_for_resolution(self.forecast_resolution_minutes)
        
        current_features = self.model_config.input_features

        return ModelConfig.get_default_config(
            model_type=model_type,
            input_resolution_minutes=self.input_resolution_minutes,
            forecast_resolution_minutes=self.forecast_resolution_minutes,
            input_features=current_features  # Pass the current feature count
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Helper method to safely get configuration values."""
        config_dict = self.__dict__
        # Check in all nested configs
        for config in [self.dataset_config, self.model_config, self.training_config]:
            if hasattr(config, key):
                return getattr(config, key)
        return config_dict.get(key, default)