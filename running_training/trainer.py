from pathlib import Path
import time
from typing import Any, Optional

from utils.logging.config import LogLevel, LoggerConfig
from utils.logging.logger import Logger
from pipeline.utils.config_utils import create_pipeline_config
from pipeline.implementations.time_series_pipeline import TimeSeriesPipeline
from models.registry.model_types import ModelType, initialize_model_registry
from models.register_models import register_models
from .progress import TrainingProgress
from .config import get_default_config_params

def train_model(data_path: str) -> None:
    """Train the transformer model for energy consumption forecasting."""
    logger = Logger.get_logger(
        __name__,
        LoggerConfig(
            level=LogLevel.INFO,
            component_name="Training",
            include_timestamp=True,
            json_output=False,
            file_path=Path("logs/training.log"),
            encoding='utf-8'  # Add UTF-8 encoding
        )
    )

    try:
        logger.info("Starting training process", {"data_path": data_path})

        # Initialize model registry
        initialize_model_registry()
        register_models()
        
        # Get configuration
        model_params, training_params, dataset_params = get_default_config_params()
        
        # Create pipeline configuration
        config = create_pipeline_config(
            data_path=data_path,
            model_type=ModelType.HOURLY_TRANSFORMER,
            model_params=model_params,
            training_params=training_params,
            dataset_params=dataset_params
        )
        
        # Initialize training progress tracker
        progress = TrainingProgress(
            total_epochs=model_params["max_epochs"],
            logger=logger
        )
        
        # Initialize and run pipeline
        pipeline = TimeSeriesPipeline(config)
        pipeline.add_progress_callback(progress.log_epoch)
        pipeline.add_batch_callback(progress.log_batch)
        
        logger.info("Starting model training...")
        experiment = pipeline.run()
        
        if experiment is None:
            raise RuntimeError("Training failed to produce experiment results")

        # Log results and save experiment
        _log_training_results(logger, experiment, progress.start_time)
        experiment.save_to_json_file()
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error("Training failed", {"error": str(e)})
        raise

def _log_training_results(logger: Logger, experiment: Any, start_time: float) -> None:
    """Log the final training results."""
    train_losses = experiment.training_report.train_losses if experiment.training_report else []
    val_losses = experiment.training_report.val_losses if experiment.training_report else []
    
    logger.info("Training Complete!", {
        "Duration": f"{time.time() - start_time:.2f}s",
        "Best Train Loss": f"{min(train_losses):.6f}",
        "Best Val Loss": f"{min(val_losses):.6f}" if val_losses else "N/A",
        "Early Stopping Epoch": experiment.training_report.early_stopping_epoch if experiment.training_report else None
    })
