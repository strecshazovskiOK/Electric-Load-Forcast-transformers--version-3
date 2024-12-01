# tests/integration/test_transformers.py
import pytest
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from data_loading.base.base_dataset import DatasetConfig
from data_loading.factories import DatasetFactory


def calculate_expected_features(config):
    """Calculate expected number of features based on configuration"""
    n_base_features = 1  # Value itself
    n_time_features = 0
    n_workday_features = 0

    if config.include_time_information:
        n_time_features = 24 + 12 if config.one_hot_time_variables else 4
        n_workday_features = 5  # Workday indicator features

    return n_base_features + n_time_features + n_workday_features

def test_transformer_configurations():
    """Test transformer dataset with different configurations"""
    configs = [
        # Basic config with default features
        {
            'window': 24,
            'horizon': 12,
            'include_time': False,
            'one_hot': False,
            'expected_features': 1  # Just the value
        },
        # Config with cyclical time features
        {
            'window': 24,
            'horizon': 12,
            'include_time': True,
            'one_hot': False,
            'expected_features': 10  # Value + cyclical time + workday
        },
        # Config with cyclical time features (ignores one-hot setting)
        {
            'window': 24,
            'horizon': 12,
            'include_time': True,
            'one_hot': True,
            'expected_features': 10  # Same as above, transformer always uses cyclical
        },
        # Different window/horizon sizes
        {
            'window': 48,
            'horizon': 24,
            'include_time': True,
            'one_hot': False,
            'expected_features': 10  # Same feature count
        }
    ]

    for cfg in configs:
        print(f"\nTesting transformer config:")
        print(f"Window: {cfg['window']}, Horizon: {cfg['horizon']}")
        print(f"Include time: {cfg['include_time']}, One-hot: {cfg['one_hot']}")
        print("Note: Transformer dataset always uses cyclical encoding")

        config = DatasetConfig(
            time_variable='timestamp',
            target_variable='value',
            time_series_window_in_hours=cfg['window'],
            forecasting_horizon_in_hours=cfg['horizon'],
            is_single_time_point_prediction=False,
            include_time_information=cfg['include_time'],
            time_series_scaler=StandardScaler(),
            is_training_set=True,
            labels_count=cfg['horizon'],
            one_hot_time_variables=cfg['one_hot']  # This setting is ignored by transformer
        )

        # Create dataset
        n_points = (cfg['window'] + cfg['horizon'] + 100) * 4  # 4 measurements per hour
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_points, freq='15min'),
            'value': np.sin(np.linspace(0, 8*np.pi, n_points))
        })

        dataset = DatasetFactory.create_dataset('transformer', df, config)

        # Get first batch
        input_seq, target_seq = dataset[0]

        print("Dataset info:")
        print(f"Dataset length: {len(dataset)}")
        print(f"Input shape: {input_seq.shape}")
        print(f"Target shape: {target_seq.shape}")

        # Time feature breakdown when enabled
        if cfg['include_time']:
            print("Feature breakdown:")
            print("  - Value: 1")
            print("  - Hour (sin/cos): 2")
            print("  - Week (sin/cos): 2")
            print("  - Workday indicators: 5")

        print(f"Expected features: {cfg['expected_features']}")
        print(f"Actual features: {input_seq.shape[1]}")

        # Check sequence lengths
        assert input_seq.shape[0] == cfg['window'], \
            f"Input sequence length should be {cfg['window']}, got {input_seq.shape[0]}"
        assert target_seq.shape[0] == cfg['horizon'], \
            f"Target sequence length should be {cfg['horizon']}, got {target_seq.shape[0]}"

        # Check feature dimensions match expected
        assert input_seq.shape[1] == cfg['expected_features'], \
            f"Expected {cfg['expected_features']} features, got {input_seq.shape[1]}"
        assert target_seq.shape[1] == cfg['expected_features'], \
            f"Expected {cfg['expected_features']} features in target, got {target_seq.shape[1]}"


def test_transformer_consistency():
    """Test transformer dataset output consistency"""
    config = DatasetConfig(
        time_variable='timestamp',
        target_variable='value',
        time_series_window_in_hours=24,
        forecasting_horizon_in_hours=12,
        is_single_time_point_prediction=False,
        include_time_information=True,
        time_series_scaler=StandardScaler(),
        is_training_set=True,
        labels_count=12,
        one_hot_time_variables=False
    )

    # Create dataset with deterministic pattern
    n_points = 1000
    timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='H')

    # Create pattern with known periodicity
    hour_of_day = timestamps.hour.values
    day_of_week = timestamps.dayofweek.values

    values = (
            1000  # Base load
            + 200 * np.sin(2 * np.pi * hour_of_day / 24)  # Daily pattern
            + 100 * np.sin(2 * np.pi * day_of_week / 7)  # Weekly pattern
            + np.random.normal(0, 10, size=n_points)  # Small noise
    )

    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })

    dataset = DatasetFactory.create_dataset('transformer', df, config)

    # Check multiple sequences
    for i in range(5):
        input_seq, target_seq = dataset[i]

        input_end_value = input_seq[-1, 0].item()  # Last value in input
        target_start_value = target_seq[0, 0].item()  # First value in target

        print(f"\nSequence {i}:")
        print(f"Input end value: {input_end_value:.4f}")
        print(f"Target start value: {target_start_value:.4f}")

        # Values should be different (they're consecutive time points)
        assert input_end_value != target_start_value, \
            f"Input and target should not overlap at sequence {i}"

def test_transformer_feature_consistency():
    """Test that transformer dataset features are consistent"""
    config = DatasetConfig(
        time_variable='timestamp',
        target_variable='value',
        time_series_window_in_hours=24,
        forecasting_horizon_in_hours=12,
        is_single_time_point_prediction=False,
        include_time_information=True,
        time_series_scaler=StandardScaler(),
        is_training_set=True,
        labels_count=12,
        one_hot_time_variables=False
    )

    # Create dataset
    n_points = 200 * 4  # 200 hours with 15-minute intervals
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_points, freq='15min'),
        'value': np.sin(np.linspace(0, 8*np.pi, n_points))
    })

    dataset = DatasetFactory.create_dataset('transformer', df, config)

    # Get multiple sequences
    sequences = [dataset[i] for i in range(3)]

    print("\nFeature consistency check:")
    for i, (input_seq, target_seq) in enumerate(sequences):
        print(f"\nSequence {i}:")
        print(f"Input features range: [{input_seq.min():.4f}, {input_seq.max():.4f}]")
        print(f"Target features range: [{target_seq.min():.4f}, {target_seq.max():.4f}]")

        # Check feature ranges are reasonable
        assert torch.all(torch.isfinite(input_seq)), \
            f"Non-finite values in input sequence {i}"
        assert torch.all(torch.isfinite(target_seq)), \
            f"Non-finite values in target sequence {i}"

        # Check time features are properly normalized
        if config.include_time_information:
            time_features = input_seq[:, 1:]  # All features except the value
            assert torch.all(torch.abs(time_features) <= 1.0), \
                f"Time features not properly normalized in sequence {i}"