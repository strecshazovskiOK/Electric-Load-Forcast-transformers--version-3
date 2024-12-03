import argparse
import time
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.preprocessing import StandardScaler
import torch

from pipeline.utils.config_utils import create_pipeline_config
from pipeline.implementations.time_series_pipeline import TimeSeriesPipeline
from models.registry.model_types import ModelType, initialize_model_registry
from models.register_models import register_models
from evaluation.evaluator import Evaluator
from evaluation.metrics import MetricConfig
from utils.logging.config import LogLevel, LoggerConfig
from utils.logging.logger import Logger


def get_default_config_params() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Get optimized configuration parameters for transformer-based energy forecasting.
    Returns a tuple of (model_params, training_params, dataset_params).
    """
    # Core architecture dimensions
    sequence_length = 24   # Hours in a day for hourly predictions
    n_features = 7        # Energy value (1) + cyclical time encodings (6)
    d_model = 512        # Transformer embedding dimension
    n_heads = 8          # Number of attention heads (must divide d_model evenly)
    
    # Compute derived parameters
    d_ff = d_model * 4   # Feed-forward dimension (standard transformer ratio)
    n_layers = 6         # Number of transformer layers (standard configuration)

    model_params = {
        # Core model identification
        "model_type": ModelType.HOURLY_TRANSFORMER,
        
        # Time resolution and sequence configuration
        "input_resolution_minutes": 60,
        "forecast_resolution_minutes": 60,
        "lookback_periods": sequence_length,
        "forecast_periods": sequence_length,
        
        # Model architecture parameters
        "input_features": n_features,
        "output_features": 1,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_encoder_layers": n_layers,
        "n_decoder_layers": n_layers,
        "d_ff": d_ff,
        "dropout": 0.1,
        "max_sequence_length": 5000,
        
        # Time series specific parameters
        "value_features": 1,
        "time_features": n_features - 1,
        "kernel_size": 3,
        "batch_first": True,
        
        # Neural network specific
        "hidden_dims": [64, 32],
        "activation": "relu",
        
        # Training
        "batch_size": 32,
        "learning_rate": 0.0001,
        "max_epochs": 100,
        "optimizer": "adamw",
        "optimizer_config": {
            "weight_decay": 0.01,
            "betas": (0.9, 0.98)
        },
        "scheduler": "one_cycle",
        "scheduler_config": {
            "pct_start": 0.3,
            "div_factor": 25.0,
            "final_div_factor": 1000.0
        },
        "criterion": "mse",
        "criterion_config": {},
        
        # Device
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    training_params = {
        # Core training settings
        "learning_rate": 0.0001,
        "max_epochs": 100,
        "batch_size": 32,
        "device": model_params["device"],
        
        # Early stopping
        "use_early_stopping": True,
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 1e-4,
        "save_best_model": True,
        
        # Sequence handling
        "input_resolution_minutes": 60,
        "forecast_resolution_minutes": 60,
        "transformer_labels_count": sequence_length,
        "forecasting_horizon": sequence_length,
        
        # Teacher forcing
        "transformer_use_teacher_forcing": True,
        "teacher_forcing_ratio": 0.5,
        "teacher_forcing_schedule": "linear_decay",
        
        # Optimization
        "gradient_clip_val": 1.0,
        "optimizer": "adamw",
        "optimizer_config": {
            "weight_decay": 0.01,
            "betas": (0.9, 0.98),
            "eps": 1e-8
        },
        
        # Learning rate scheduling
        "scheduler": "one_cycle",
        "scheduler_config": {
            "pct_start": 0.3,
            "div_factor": 25.0,
            "final_div_factor": 1000.0,
            "anneal_strategy": "cos",
            "cycle_momentum": True
        },
        
        # Validation settings
        "validation_frequency": 1,         # Validate every N epochs
        "validation_metric": "mae",        # Primary metric for model selection
        
        # Logging and checkpointing
        "log_frequency": 10,               # Log every N batches
        "checkpoint_frequency": 5,         # Save checkpoint every N epochs
        "keep_last_n_checkpoints": 3       # Number of checkpoints to keep
    }

    dataset_params = {
        # Core parameters
        "time_variable": "utc_timestamp",
        "target_variable": "energy_consumption",
        
        # Resolution and sequence settings
        "input_resolution_minutes": 60,
        "forecast_resolution_minutes": 60,
        "lookback_periods": sequence_length,
        "forecast_periods": sequence_length,
        
        # Core feature configuration
        "is_single_time_point_prediction": False,
        "include_time_information": True,
        "is_training_set": True,
        "labels_count": sequence_length,
        "one_hot_time_variables": False,
        
        # Data preprocessing
        "normalize_data": True,
        "scaling_method": "standard",
        "time_series_scaler": StandardScaler(),
        "handle_missing_values": "interpolate",
        "remove_outliers": True,
        "outlier_std_threshold": 3.0,
        
        # Basic feature flags
        "add_time_features": True,
        "add_holiday_features": True,
        "add_weather_features": False,
        
        # Detailed time features
        "add_hour_feature": True,
        "add_weekday_feature": True,
        "add_month_feature": True,
        "add_season_feature": False,
        "add_year_feature": False,
        
        # Data augmentation
        "use_data_augmentation": False,
        "augmentation_methods": ["jitter", "scaling"],
        "augmentation_probability": 0.3,
        
        # Sequence handling
        "padding_value": 0.0,
        "mask_padding": True,
        "max_sequence_gaps": 3
    }

    return model_params, training_params, dataset_params

def print_epoch_summary(epoch: int, total_epochs: int, train_loss: float, 
                       val_loss: float, best_train: float, best_val: float,
                       time_elapsed: float) -> None:
    """Print a formatted summary of the current training epoch."""
    print(f"\nEpoch [{epoch}/{total_epochs}] - Time: {time_elapsed:.2f}s")
    print(f"Training Loss: {train_loss:.4f} (Best: {best_train:.4f})")
    if val_loss is not None:
        print(f"Validation Loss: {val_loss:.4f} (Best: {best_val:.4f})")
        train_improvement = ((train_loss - best_train) / train_loss) * 100
        val_improvement = ((val_loss - best_val) / val_loss) * 100
        print(f"Improvements - Train: {train_improvement:.2f}%, Val: {val_improvement:.2f}%")

def train_model(data_path: str) -> None:
    """
    Train the transformer model for energy consumption forecasting with enhanced progress monitoring.
    
    Args:
        data_path: Path to the CSV data file containing energy consumption data
    """
    try:
        # Initialize logging
        logger = Logger.get_logger(
            __name__,
            LoggerConfig(
                level=LogLevel.INFO,
                component_name="TrainingPipeline",
                include_timestamp=True,
                json_output=True,
                file_path=Path("logs/training.log")
            )
        )
        logger.info("Starting training process", {"data_path": data_path})

        # Initialize systems
        initialize_model_registry()
        register_models()
        
        # Get and validate configuration
        model_params, training_params, dataset_params = get_default_config_params()
        
        # Create pipeline configuration
        config = create_pipeline_config(
            data_path=data_path,
            model_type=ModelType.HOURLY_TRANSFORMER,
            model_params=model_params,
            training_params=training_params,
            dataset_params=dataset_params
        )
        
        # Initialize evaluator for monitoring training progress
        evaluator = Evaluator(
            scaler=config.dataset_config.time_series_scaler,
            metric_config=MetricConfig(
                resolution_minutes=config.dataset_config.forecast_resolution_minutes
            ),
            resolution_minutes=config.dataset_config.forecast_resolution_minutes
        )
        
        # Initialize and run pipeline
        print("\nInitializing training pipeline...")
        pipeline = TimeSeriesPipeline(config)
        
        print("\nStarting model training...")
        start_time = time.time()
        
        # Set up progress tracking
        display_frequency = max(1, model_params["max_epochs"] // 20)  # Show ~20 updates
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        
        def progress_callback(epoch: int, train_loss: float, val_loss: float) -> None:
            """Callback to monitor training progress."""
            nonlocal best_train_loss, best_val_loss
            
            best_train_loss = min(best_train_loss, train_loss)
            if val_loss is not None:
                best_val_loss = min(best_val_loss, val_loss)
            
            if epoch % display_frequency == 0 or epoch == model_params["max_epochs"] - 1:
                time_elapsed = time.time() - start_time
                print_epoch_summary(
                    epoch + 1,  # Convert to 1-based indexing for display
                    model_params["max_epochs"],
                    train_loss,
                    val_loss,
                    best_train_loss,
                    best_val_loss,
                    time_elapsed
                )
                
                # Log detailed metrics
                logger.info("Training progress", {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss) if val_loss is not None else None,
                    "best_train_loss": float(best_train_loss),
                    "best_val_loss": float(best_val_loss),
                    "time_elapsed": time_elapsed
                })
        
        # Add callback to pipeline
        pipeline.add_progress_callback(progress_callback)
        
        # Run training
        experiment = pipeline.run()
        
        if experiment is None:
            raise RuntimeError("Training failed to produce experiment results")
        
        # Calculate final metrics
        final_metrics = {}
        if hasattr(experiment.evaluation, 'total_metrics'):
            final_metrics = experiment.evaluation.total_metrics
            logger.info("Final model metrics", final_metrics)
        
        # Print final summary
        print("\nTraining Complete!")
        print(f"Total Duration: {time.time() - start_time:.2f} seconds")
        print(f"Inference Time: {experiment.test_time:.2f} seconds")
        
        print("\nFinal Model Performance:")
        for metric, value in final_metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        if experiment.training_report:
            train_losses = experiment.training_report.train_losses
            val_losses = experiment.training_report.val_losses
            
            print("\nTraining Summary:")
            print(f"Starting Train Loss: {train_losses[0]:.4f}")
            print(f"Final Train Loss: {train_losses[-1]:.4f}")
            print(f"Best Train Loss: {min(train_losses):.4f}")
            
            if val_losses:
                print(f"\nStarting Val Loss: {val_losses[0]:.4f}")
                print(f"Final Val Loss: {val_losses[-1]:.4f}")
                print(f"Best Val Loss: {min(val_losses):.4f}")
            
            if experiment.training_report.early_stopping_epoch:
                print(f"\nEarly stopping occurred at epoch {experiment.training_report.early_stopping_epoch}")
        
        # Save experiment
        logger.info("Saving experiment results...")
        experiment.save_to_json_file()
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error("Training failed", {"error": str(e)})
        raise

def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description='Train transformer model for energy consumption forecasting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to the CSV data file containing energy consumption data'
    )
    
    # Parse arguments and train
    args = parser.parse_args()
    train_model(args.data)

if __name__ == '__main__':
    main()