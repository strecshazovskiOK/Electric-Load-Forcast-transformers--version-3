# tests/integration/test_data_pipeline_advanced.py
from __future__ import annotations
from typing import Dict, Tuple, Any, Union, Optional
from datetime import datetime, date
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing_extensions import Protocol

from data_loading.base.base_dataset import DatasetConfig
from data_loading.factories import DatasetFactory
from data_loading.loaders.time_series_loader import TimeInterval, TimeSeriesLoader


def test_different_prediction_horizons(dataset_config: DatasetConfig):
    """Test different prediction horizon configurations"""
    horizons = [1, 6, 12, 24]

    for horizon in horizons:
        print(f"\nTesting horizon {horizon}:")
        # Create configuration with specific horizon
        config = DatasetConfig(
            time_variable=dataset_config.time_variable,
            target_variable=dataset_config.target_variable,
            time_series_window_in_hours=24,
            forecasting_horizon_in_hours=horizon,
            is_single_time_point_prediction=True,
            include_time_information=True,
            time_series_scaler=StandardScaler(),
            is_training_set=True,
            labels_count=0,
            one_hot_time_variables=False
        )

        # Create larger dataset to ensure enough points
        n_points = (24 * 4 + horizon * 4 + 400)  # Ensure enough points for window and horizon
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_points, freq='15min'),
            'value': np.sin(np.linspace(0, 8 * np.pi, n_points, dtype=np.float64).tolist())  # Simple sine wave # type: ignore
        })

        print(f"Number of data points: {n_points}")
        print("Data frequency: 15min")

        dataset = DatasetFactory.create_dataset('standard', df, config)

        assert dataset.prepared_time_series_input is not None, "Input tensor is None"
        assert dataset.prepared_time_series_target is not None, "Target tensor is None"

        print(f"Input tensor shape: {dataset.prepared_time_series_input.shape}")
        print(f"Target tensor shape: {dataset.prepared_time_series_target.shape}")

        # Check target dimension
        target_size = 1 if config.is_single_time_point_prediction else horizon
        assert dataset.prepared_time_series_target.shape[-1] == target_size, \
            f"Expected target size {target_size}, got {dataset.prepared_time_series_target.shape[-1]}"

def test_transformer_dataset_creation(sample_timeseries_data, time_intervals):
    """Test creation of transformer dataset"""
    window_size = 24
    horizon = 12

    config = DatasetConfig(
        time_variable='timestamp',
        target_variable='value',
        time_series_window_in_hours=window_size,
        forecasting_horizon_in_hours=horizon,
        is_single_time_point_prediction=False,  # Transformer predicts sequence
        include_time_information=True,
        time_series_scaler=StandardScaler(),
        is_training_set=True,
        labels_count=horizon,  # Must match forecasting_horizon_in_hours
        one_hot_time_variables=True
    )

    loader = TimeSeriesLoader('timestamp', 'value')
    df = loader.load(sample_timeseries_data)  # Use fixture directly

    # Ensure enough data points
    min_required = (window_size + horizon) * 4  # 4 measurements per hour
    if len(df) < min_required:
        print(f"\nWarning: Dataset too small. Creating larger dataset.")
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=min_required*2, freq='15min'),
            'value': np.sin(np.linspace(0, 8*np.pi, min_required*2)) # type: ignore
        })

    dataset = DatasetFactory.create_dataset('transformer', df, config)
    
    # Add null checks before accessing .shape
    assert dataset.prepared_time_series_input is not None, "Input tensor is None"
    assert dataset.prepared_time_series_target is not None, "Target tensor is None"

    print("\nTransformer Dataset Info:")
    print(f"Dataset length: {len(dataset)}")
    print(f"Config time window: {config.time_series_window_in_hours}")
    print(f"Config forecast horizon: {config.forecasting_horizon_in_hours}")
    print(f"Config labels count: {config.labels_count}")

    input_seq, target_seq = dataset[0]
    print(f"Input sequence shape: {input_seq.shape}")
    print(f"Target sequence shape: {target_seq.shape}")

    # Basic assertions
    assert dataset.rows is not None, "Rows tensor is None"
    assert len(dataset) > 0, "Dataset is empty"
    assert input_seq.dim() == 2, "Input sequence should be 2D"
    assert target_seq.dim() == 2, "Target sequence should be 2D"

    # Check sequence lengths
    assert input_seq.shape[0] == window_size, \
        f"Input sequence length should be {window_size}, got {input_seq.shape[0]}"
    assert target_seq.shape[0] == horizon, \
        f"Target sequence length should be {horizon}, got {target_seq.shape[0]}"

def test_one_hot_time_encoding(sample_timeseries_data, dataset_config):
    """Test one-hot encoding of time features"""
    config = DatasetConfig(
        time_variable=dataset_config.time_variable,
        target_variable=dataset_config.target_variable,
        time_series_window_in_hours=24,
        forecasting_horizon_in_hours=12,
        is_single_time_point_prediction=True,
        include_time_information=True,
        time_series_scaler=StandardScaler(),
        is_training_set=True,
        labels_count=None,
        one_hot_time_variables=True  # Enable one-hot encoding
    )

    loader = TimeSeriesLoader('timestamp', 'value')
    df = loader.load(sample_timeseries_data)  # Use fixture directly
    dataset = DatasetFactory.create_dataset('standard', df, config)

    # Add null check before accessing shape
    assert dataset.prepared_time_series_input is not None, "Input tensor is None"
    
    # Calculate expected feature dimensions
    base_features = config.time_series_window_in_hours
    hour_features = 24  # One-hot encoded hours
    month_features = 12  # One-hot encoded months
    workday_features = 5  # Workday indicator features
    expected_features = base_features + hour_features + month_features + workday_features

    print("\nOne-hot encoding test:")
    print(f"Config: {config}")
    print(f"Input features shape: {dataset.prepared_time_series_input.shape}")
    print(f"Expected features: {expected_features}")
    print("Feature breakdown:")
    print(f"  Base features: {base_features}")
    print(f"  Hour features: {hour_features}")
    print(f"  Month features: {month_features}")
    print(f"  Workday features: {workday_features}")

    assert dataset.prepared_time_series_input.shape[1] == expected_features, \
        f"Expected {expected_features} features, got {dataset.prepared_time_series_input.shape[1]}"

def test_data_consistency(
    sample_timeseries_data, 
    dataset_config: DatasetConfig, 
    time_intervals: dict[str, tuple[date, date]]  # Changed from datetime.date to date
) -> None:
    """Test data consistency across train/val/test splits"""
    loader = TimeSeriesLoader('timestamp', 'value')
    
    # Create sample data that covers all intervals
    total_days = (time_intervals['test'][1] - time_intervals['train'][0]).days + 1
    n_points = total_days * 24 * 4  # 4 points per hour
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(
            start=time_intervals['train'][0],
            periods=n_points, 
            freq='15min'
        ),
        'value': np.sin(np.linspace(0, 8*np.pi, n_points))
    })
    
    # Set up intervals with correct parameter names
    train_interval = TimeInterval(
        min_date=time_intervals['train'][0],
        max_date=time_intervals['train'][1]
    )
    val_interval = TimeInterval(
        min_date=time_intervals['val'][0],
        max_date=time_intervals['val'][1]
    )
    test_interval = TimeInterval(
        min_date=time_intervals['test'][0],
        max_date=time_intervals['test'][1]
    )
    
    # Split and verify data
    train_df, val_df, test_df = loader.split(df, train_interval, val_interval, test_interval)
    
    # Verify we have data in all splits
    assert len(train_df) > 0, "Train dataset is empty"
    assert len(val_df) > 0, "Validation dataset is empty"
    assert len(test_df) > 0, "Test dataset is empty"

    # Create datasets
    train_dataset = DatasetFactory.create_dataset('standard', train_df, dataset_config)
    val_config = DatasetConfig(**{**dataset_config.__dict__, 'is_training_set': False})
    test_config = DatasetConfig(**{**dataset_config.__dict__, 'is_training_set': False})

    val_dataset = DatasetFactory.create_dataset('standard', val_df, val_config)
    test_dataset = DatasetFactory.create_dataset('standard', test_df, test_config)

    # Add null checks
    assert train_dataset.prepared_time_series_input is not None, "Train input is None"
    assert val_dataset.prepared_time_series_input is not None, "Val input is None"
    assert test_dataset.prepared_time_series_input is not None, "Test input is None"

    print("\nData consistency test:")
    print(f"Train set shape: {train_dataset.prepared_time_series_input.shape}")
    print(f"Val set shape: {val_dataset.prepared_time_series_input.shape}")
    print(f"Test set shape: {test_dataset.prepared_time_series_input.shape}")

    # Check consistent feature dimensions
    assert train_dataset.prepared_time_series_input.shape[1] == \
           val_dataset.prepared_time_series_input.shape[1] == \
           test_dataset.prepared_time_series_input.shape[1], \
        "Inconsistent feature dimensions across datasets"

    # Check scaling consistency using assert-guarded calls to std_mean()
    if all(x is not None for x in [train_dataset.prepared_time_series_input,
                                  val_dataset.prepared_time_series_input,
                                  test_dataset.prepared_time_series_input]):
        train_stats = torch.std_mean(train_dataset.prepared_time_series_input)
        val_stats = torch.std_mean(val_dataset.prepared_time_series_input)  
        test_stats = torch.std_mean(test_dataset.prepared_time_series_input)

        print("\nData statistics:")
        print(f"Train - mean: {train_stats[1]:.4f}, std: {train_stats[0]:.4f}")
        print(f"Val   - mean: {val_stats[1]:.4f}, std: {val_stats[0]:.4f}")
        print(f"Test  - mean: {test_stats[1]:.4f}, std: {test_stats[0]:.4f}")

        # Check for reasonable consistency in statistics
        mean_threshold = 0.5
        std_threshold = 0.5

        assert abs(train_stats[1] - val_stats[1]) < mean_threshold, \
            f"Mean difference between train and val ({abs(train_stats[1] - val_stats[1]):.4f}) exceeds threshold ({mean_threshold})"
        assert abs(train_stats[0] - val_stats[0]) < std_threshold, \
            f"Std difference between train and val ({abs(train_stats[0] - val_stats[0])::.4f}) exceeds threshold ({std_threshold})"