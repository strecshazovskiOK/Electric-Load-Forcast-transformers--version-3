# pipeline/utils/config_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from data_loading.base.base_dataset import DatasetConfig
from models.config.model_config import ModelConfig
from models.registry.model_types import ModelType
from pipeline.config.pipeline_config import PipelineConfig  # Fix import path
from training.config.training_config import (  # Fix import path
    BaseConfig,
    TransformerTrainingConfig,
    ResolutionBasedTrainingConfig
)

def get_resolution_parameters(
    input_resolution: int,
    forecast_resolution: int
) -> Dict[str, Any]:
    """
    Calculate appropriate parameters based on input and forecast resolutions.
    This helps ensure consistent configuration across different components.
    """
    # Calculate points per period for different time scales
    points_per_hour = 60 // min(input_resolution, 60)
    points_per_day = 24 * points_per_hour
    
    # Determine appropriate window sizes based on resolution
    if forecast_resolution <= 60:  # Hourly or sub-hourly
        lookback_hours = 24  # One day of history
        forecast_hours = 12  # 12 hours ahead
    elif forecast_resolution <= 1440:  # Daily
        lookback_hours = 168  # One week of history
        forecast_hours = 72  # 3 days ahead
    else:  # Monthly
        lookback_hours = 720  # 30 days of history
        forecast_hours = 168  # 1 week ahead

    return {
        'input_resolution_minutes': input_resolution,
        'forecast_resolution_minutes': forecast_resolution,
        'lookback_periods': lookback_hours * (60 // input_resolution),
        'forecast_periods': forecast_hours * (60 // forecast_resolution),
        'points_per_hour': points_per_hour,
    }

def create_pipeline_config(
    data_path: str | Path,
    model_type: ModelType,
    model_params: Dict[str, Any],
    training_params: Dict[str, Any],
    dataset_params: Dict[str, Any]
) -> PipelineConfig:
    """Create pipeline configuration with proper feature dimension handling."""
    print("\nDEBUG: Creating Pipeline Configuration")
    print(f"Initial model params - input_features: {model_params.get('input_features')}")
    
    # Extract and validate feature dimensions
    n_features = model_params['input_features']
    value_features = model_params['value_features']
    time_features = model_params['time_features']
    
    # Validate feature dimensions match
    assert n_features == value_features + time_features, (
        f"Feature dimension mismatch! Total ({n_features}) must equal "
        f"value ({value_features}) + time ({time_features})"
    )
    
    # Create configs with explicit feature dimensions
    dataset_config = DatasetConfig(**dataset_params)
    
    # Ensure model config preserves feature dimensions
    model_config = ModelConfig(**{
        **model_params,
        'input_features': n_features,
        'value_features': value_features,
        'time_features': time_features
    })
    
    # Create training config
    training_config = create_training_config(
        model_type=model_type,
        params=training_params,
        input_resolution=dataset_params['input_resolution_minutes'],
        forecast_resolution=dataset_params['forecast_resolution_minutes']
    )
    
    print(f"Final model config - input_features: {model_config.input_features}")
    
    return PipelineConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        training_config=training_config,
        data_path=Path(data_path),
        input_resolution_minutes=dataset_params['input_resolution_minutes'],
        forecast_resolution_minutes=dataset_params['forecast_resolution_minutes']
    )
def create_training_config(
    model_type: ModelType,
    params: Dict[str, Any],
    input_resolution: int,
    forecast_resolution: int
) -> BaseConfig:  # Changed return type to BaseConfig
    """
    Create an appropriate training configuration based on model type and resolution.
    This function handles the complexity of choosing and configuring the right training setup.
    """
    # Remove resolution parameters from params if they exist
    filtered_params = params.copy()
    filtered_params.pop('input_resolution_minutes', None)
    filtered_params.pop('forecast_resolution_minutes', None)
    
    # Adjust batch size based on resolution
    if 'batch_size' not in filtered_params:
        filtered_params['batch_size'] = _get_default_batch_size(forecast_resolution)

    # Adjust learning rate based on resolution
    if 'learning_rate' not in filtered_params:
        filtered_params['learning_rate'] = _get_default_learning_rate(forecast_resolution)

    if model_type.is_transformer:
        # For transformer models, use resolution-aware transformer config
        filtered_params = filter_config_params(filtered_params, TransformerTrainingConfig)
        return TransformerTrainingConfig(
            input_resolution_minutes=input_resolution,
            forecast_resolution_minutes=forecast_resolution,
            **filtered_params
        )
    else:
        # For other models, use base config
        filtered_params = filter_config_params(filtered_params, BaseConfig)  # Changed to BaseConfig
        return BaseConfig(**filtered_params)  # Changed to BaseConfig

def _get_default_batch_size(resolution: int) -> int:
    """Calculate appropriate batch size based on resolution."""
    if resolution <= 15:
        return 64  # Larger batches for high-frequency data
    elif resolution <= 60:
        return 32  # Medium batches for hourly data
    elif resolution <= 1440:
        return 16  # Smaller batches for daily data
    else:
        return 8   # Smallest batches for monthly data

def _get_default_learning_rate(resolution: int) -> float:
    """Calculate appropriate learning rate based on resolution."""
    if resolution <= 15:
        return 0.001  # Smaller learning rate for high-frequency data
    elif resolution <= 60:
        return 0.002  # Medium learning rate for hourly data
    elif resolution <= 1440:
        return 0.005  # Larger learning rate for daily data
    else:
        return 0.01  # Largest learning rate for monthly data

def filter_config_params(params: Dict[str, Any], config_class: Any) -> Dict[str, Any]:
    """Filter configuration parameters to match the target class fields."""
    return {
        k: v for k, v in params.items() 
        if k in config_class.__dataclass_fields__  # type: ignore
    }