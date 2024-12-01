# tests/integration/test_data_pipeline.py
import itertools
import pytest
import pandas as pd
import torch
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from data_loading.base.base_dataset import DatasetConfig
from data_loading.factories import DatasetFactory
from data_loading.loaders.time_series_loader import TimeInterval, TimeSeriesLoader
from data_loading.preprocessing.data_transformer import DataTransformer

from typing import Optional

def print_tensor_stats(name: str, tensor: Optional[torch.Tensor]) -> None:
    if tensor is not None:
        print(f"{name} stats:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Mean: {tensor.mean().item():.4f}")
        print(f"  Std: {tensor.std().item():.4f}")
        print(f"  Min: {tensor.min().item():.4f}")
        print(f"  Max: {tensor.max().item():.4f}")
    else:
        print(f"{name} is None")

@pytest.fixture
def sample_data_csv(tmp_path):
    # Create sample data with exactly 4 measurements per hour
    dates = []
    values = []
    start_date = pd.Timestamp('2024-01-01')

    # Create a more realistic load pattern
    base_load = 1000  # Base load
    for hour, quarter in itertools.product(range(1000), range(4)):
        timestamp = start_date + pd.Timedelta(hours=hour) + pd.Timedelta(minutes=15*quarter)
        hour_of_day = timestamp.hour
        day_of_week = timestamp.dayofweek

        hour_pattern = 200 * np.sin(2 * np.pi * (hour_of_day + quarter/4) / 24)  # Daily pattern
        week_pattern = -100 if day_of_week >= 5 else 100  # Weekend pattern
        noise = np.random.normal(0, 20)  # Random noise

        dates.append(timestamp)
        values.append(base_load + hour_pattern + week_pattern + noise)

    df = pd.DataFrame({
        'timestamp': dates,
        'value': values
    })

    print("\nCreated sample data:")
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Sample values:\n{df.head()}")

    csv_path = tmp_path / "test_timeseries.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def dataset_config():
    """Create a dataset configuration for testing"""
    return DatasetConfig(
        time_variable='timestamp',
        target_variable='value',
        time_series_window_in_hours=24,
        forecasting_horizon_in_hours=12,
        is_single_time_point_prediction=True,
        include_time_information=True,
        time_series_scaler=StandardScaler(),
        is_training_set=True,
        labels_count=None,
        one_hot_time_variables=False
    )

@pytest.fixture
def time_intervals():
    """Create time intervals for train/validation/test splits"""
    train_start = date(2024, 1, 1)
    train_end = date(2024, 1, 20)
    val_start = date(2024, 1, 21)
    val_end = date(2024, 1, 25)
    test_start = date(2024, 1, 26)
    test_end = date(2024, 1, 31)

    return {
        'train': TimeInterval(train_start, train_end),
        'val': TimeInterval(val_start, val_end),
        'test': TimeInterval(test_start, test_end)
    }

def test_complete_standard_pipeline(sample_data_csv, dataset_config, time_intervals):
    # 1. Load and verify data
    loader = TimeSeriesLoader('timestamp', 'value')
    df = loader.load(sample_data_csv)
    print("\n=== Initial Data ===")
    print(f"Total measurements: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Measurements per hour: {len(df) / ((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600):.2f}")
    print(f"Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")

    # 2. Split data and verify splits
    train_df, val_df, test_df = loader.split(
        df,
        time_intervals['train'],
        time_intervals['val'],
        time_intervals['test']
    )

    print("\n=== Split Data ===")
    print(f"Train set: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"Val set: {len(val_df)} rows ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
    print(f"Test set: {len(test_df)} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

    # Debug raw data before transformation
    print("\n=== Raw Data Statistics ===")
    for name, data in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"{name} set:")
        print(f"  Rows: {len(data)}")
        print(f"  Value mean: {data['value'].mean():.2f}")
        print(f"  Value std: {data['value'].std():.2f}")
        print(f"  Timestamps per hour: {len(data) / ((data['timestamp'].max() - data['timestamp'].min()).total_seconds() / 3600):.2f}")

    # 3. Create training dataset
    print("\n=== Creating Training Dataset ===")
    train_dataset = DatasetFactory.create_dataset('standard', train_df, dataset_config)

    # Debug training dataset
    print_tensor_stats("Training input", train_dataset.prepared_time_series_input)
    print_tensor_stats("Training target", train_dataset.prepared_time_series_target)

    # Get fitted scaler
    fitted_scaler = train_dataset.config.time_series_scaler
    if fitted_scaler and hasattr(fitted_scaler, 'mean_'):
        print("\nScaler statistics:")
        if fitted_scaler.mean_ is not None:
            print(f"Mean: {fitted_scaler.mean_[:5]}")
        if fitted_scaler.scale_ is not None:
            print(f"Scale: {fitted_scaler.scale_[:5]}")

    # 4. Create validation dataset
    print("\n=== Creating Validation Dataset ===")
    val_config = DatasetConfig(
        time_variable=dataset_config.time_variable,
        target_variable=dataset_config.target_variable,
        time_series_window_in_hours=dataset_config.time_series_window_in_hours,
        forecasting_horizon_in_hours=dataset_config.forecasting_horizon_in_hours,
        is_single_time_point_prediction=dataset_config.is_single_time_point_prediction,
        include_time_information=dataset_config.include_time_information,
        time_series_scaler=fitted_scaler,
        is_training_set=False,
        labels_count=dataset_config.labels_count,
        one_hot_time_variables=dataset_config.one_hot_time_variables
    )

    # Debug validation data before processing
    transformer = DataTransformer()
    val_raw_data = np.array(val_df[val_config.target_variable])
    val_averaged = transformer.average_by_window(val_raw_data)
    print(f"\nValidation data before processing:")
    print(f"Raw shape: {val_raw_data.shape}")
    print(f"Averaged shape: {val_averaged.shape}")
    print(f"Raw mean: {val_raw_data.mean():.2f}")
    print(f"Averaged mean: {val_averaged.mean():.2f}")

    # Create validation dataset
    val_dataset = DatasetFactory.create_dataset('standard', val_df, val_config)

    # Debug validation dataset
    print_tensor_stats("Validation input", val_dataset.prepared_time_series_input)
    print_tensor_stats("Validation target", val_dataset.prepared_time_series_target)

    # Assertions with detailed messages
    train_input = train_dataset.prepared_time_series_input
    val_input = val_dataset.prepared_time_series_input
    
    # Get shapes safely with null checks
    input_shape = train_input.shape if train_input is not None else None 
    val_input_shape = val_input.shape if val_input is not None else None
    
    assert train_input is not None and len(train_dataset) > 0, \
        f"Training dataset is empty (input shape: {input_shape})"
    assert val_input is not None and len(val_dataset) > 0, \
        f"Validation dataset is empty (input shape: {val_input_shape})"