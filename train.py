# path: pipeline/implementations/time_series_pipeline.py
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.preprocessing import StandardScaler

from pipeline.utils.config_utils import create_pipeline_config
from pipeline.implementations.time_series_pipeline import TimeSeriesPipeline
from models.registry.model_types import ModelType, initialize_model_registry

def get_default_config_params() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Get default configuration parameters optimized for 15-min interval data."""
    model_params = {
        "input_features": 10,
        "output_features": 10,
        "d_model": 256,
        "n_heads": 8,
        "n_encoder_layers": 4,
        "n_decoder_layers": 4,
        "d_ff": 1024,
        "dropout": 0.2,
        "max_seq_length": 96,
        "transformer_labels_count": 48
    }

    training_params = {
        "learning_rate": 0.0005,
        "max_epochs": 150,
        "use_early_stopping": True,
        "early_stopping_patience": 15,
        "batch_size": 64,
        "device": "cuda",
        "transformer_labels_count": 48,
        "forecasting_horizon": 48,
        "transformer_use_teacher_forcing": True,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "optimizer_config": {
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-8
        },
        "scheduler": "cosine",
        "scheduler_config": {
            "T_max": 150,
            "eta_min": 1e-6
        }
    }

    dataset_params = {
        "time_variable": "utc_timestamp",
        "target_variable": "energy_consumption",
        "time_series_window_in_hours": 24,
        "forecasting_horizon_in_hours": 12,
        "is_single_time_point_prediction": False,
        "include_time_information": True,
        "is_training_set": True,
        "labels_count": 48,
        "one_hot_time_variables": False,
        "normalize_data": True,
        "scaling_method": "standard",
        "time_series_scaler": StandardScaler(),
        "time_resolution_minutes": 15,  # We only specify the resolution
        "add_time_features": True,
        "add_holiday_features": True,
        "add_weather_features": False
    }

    return model_params, training_params, dataset_params

def train_model(data_path: str) -> None:
    """Train the transformer model using the pipeline."""
    try:
        # Initialize model registry
        initialize_model_registry()
        
        # Get configuration parameters
        model_params, training_params, dataset_params = get_default_config_params()
        
        # Create pipeline configuration
        config = create_pipeline_config(
            data_path=data_path,
            model_type=ModelType.VANILLA_TRANSFORMER,
            model_params=model_params,
            training_params=training_params,
            dataset_params=dataset_params
        )
        
        # Initialize and run pipeline
        pipeline = TimeSeriesPipeline(config)
        experiment = pipeline.run()
        
        if experiment is None:
            raise RuntimeError("Training failed to produce results")
        
        # Print results
        print("\nTraining Complete!")
        print(f"Training time: {experiment.training_time:.2f} seconds")
        print(f"Test time: {experiment.test_time:.2f} seconds")
        
        print("\nEvaluation Metrics:")
        for metric, value in experiment.evaluation.total_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Print training history
        print("\nTraining History:")
        if experiment.training_report and hasattr(experiment.training_report, 'train_losses'):
            final_train_loss = experiment.training_report.train_losses[-1]
            print(f"Final Training Loss: {final_train_loss:.4f}")
        else:
            print("Training loss data not available")

        if experiment.training_report and hasattr(experiment.training_report, 'val_losses'):
            final_val_loss = experiment.training_report.val_losses[-1]
            print(f"Final Validation Loss: {final_val_loss:.4f}")
        else:
            print("Validation loss data not available")
        
        # Save the experiment
        experiment.save_to_json_file()
        print(f"\nExperiment saved successfully")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train transformer model for 15-min interval energy forecasting')
    parser.add_argument('--data', type=str, required=True, help='Path to the CSV data file')
    args = parser.parse_args()
    
    train_model(args.data)

if __name__ == '__main__':
    main()