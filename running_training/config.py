import torch
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from models.registry.model_types import ModelType

def get_default_config_params() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Get optimized configuration parameters for transformer-based energy forecasting."""
# Core architecture dimensions
    sequence_length = 24   # Hours in a day for hourly predictions
    n_features = 7        # Energy value (1) + cyclical time encodings (6)
    d_model = 512        # Transformer embedding dimension
    n_heads = 8          # Number of attention heads
    
    # Compute derived parameters
    d_ff = d_model * 4   # Feed-forward dimension
    n_layers = 6         # Number of transformer layers
    
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
        "batch_size": 512,
        "learning_rate": 0.00001,
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
