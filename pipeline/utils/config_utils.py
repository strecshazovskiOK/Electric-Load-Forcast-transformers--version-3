# pipeline/utils/config_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from data_loading.base.base_dataset import DatasetConfig
from models.config.model_config import ModelConfig
from models.registry.model_types import ModelType

from ..config.pipeline_config import PipelineConfig
from training.config import TrainingConfig, TransformerTrainingConfig  # Add import

def create_pipeline_config(
        data_path: str | Path,
        model_type: ModelType,
        model_params: Dict[str, Any],
        training_params: Dict[str, Any],
        dataset_params: Dict[str, Any]
) -> PipelineConfig:
    """Create a pipeline configuration from parameters."""

    # Create dataset config
    dataset_config = DatasetConfig(**dataset_params)

    # Create model config
    model_config = ModelConfig(
        model_type=model_type,  # Pass ModelType directly
        **model_params
    )

    # Choose appropriate training config based on model type
    if model_type.is_transformer:
        training_config = TransformerTrainingConfig(**training_params)
    else:
        # Remove transformer-specific parameters for non-transformer models
        base_params = {k: v for k, v in training_params.items() 
                    if not k.startswith('transformer_')}
        training_config = TrainingConfig(**base_params)

    # Create pipeline config
    return PipelineConfig(
        dataset_config=dataset_config,
        model_config=model_config,
        training_config=training_config,
        data_path=Path(data_path)
    )