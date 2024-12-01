# tests/unit/test_base_dataset.py
import pytest
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_loading.base.base_dataset import BaseDataset, DatasetConfig


class TestDataset(BaseDataset):
    """Concrete implementation of BaseDataset for testing"""
    def __init__(self, df: pd.DataFrame, config: DatasetConfig):
        super().__init__(df, config)
        self._prepare_time_series_data()  # Explicitly call prepare data

    def _prepare_time_series_data(self) -> None:
        # Calculate expected number of samples
        n_samples = len(self._df) - self.config.time_series_window_in_hours - self.config.forecasting_horizon_in_hours
        if n_samples <= 0:
            n_samples = 1  # Minimum length for testing

        # Calculate input features
        n_features = self.config.time_series_window_in_hours
        if self.config.include_time_information:
            n_features += 4  # Add time features (e.g., hour, day, month, workday)

        # Create sample input data
        self.prepared_time_series_input = torch.zeros((n_samples, n_features), dtype=torch.float32)
        for i in range(n_samples):
            # Fill with sample values (e.g., sequential numbers)
            self.prepared_time_series_input[i] = torch.arange(n_features, dtype=torch.float32)

        # Create sample target data
        target_size = 1 if self.config.is_single_time_point_prediction else self.config.forecasting_horizon_in_hours
        self.prepared_time_series_target = torch.zeros((n_samples, target_size), dtype=torch.float32)
        for i in range(n_samples):
            # Fill with sample values
            self.prepared_time_series_target[i] = torch.ones(target_size, dtype=torch.float32) * i

        # Set time labels
        self.time_labels = self._df[self.config.time_variable].iloc[:n_samples].values

    def __len__(self) -> int:
        if self.prepared_time_series_input is None:
            return 0
        return len(self.prepared_time_series_input)

    def __getitem__(self, index):
        if self.prepared_time_series_input is None or self.prepared_time_series_target is None:
            raise ValueError("Dataset not properly initialized")
        return (
            self.prepared_time_series_input[index],
            self.prepared_time_series_target[index]
        )

@pytest.fixture
def sample_config():
    return DatasetConfig(
        time_variable='timestamp',
        target_variable='value',
        time_series_window_in_hours=24,
        forecasting_horizon_in_hours=12,
        is_single_time_point_prediction=True,
        include_time_information=True,
        time_series_scaler=StandardScaler(),
        is_training_set=True,
        labels_count=0,
        one_hot_time_variables=False
    )

@pytest.fixture
def sample_df():
    # Create sample data with enough points for window and horizon
    dates = pd.date_range(start='2024-01-01', periods=48, freq='H')
    values = np.sin(np.linspace(0, 4*np.pi, 48))  # Create sinusoidal pattern
    return pd.DataFrame({
        'timestamp': dates,
        'value': values
    })

def test_dataset_initialization(sample_df, sample_config):
    dataset = TestDataset(sample_df, sample_config)
    assert isinstance(dataset, BaseDataset)
    assert isinstance(dataset._df, pd.DataFrame)
    assert dataset.config == sample_config
    assert dataset.prepared_time_series_input is not None
    assert dataset.prepared_time_series_target is not None

def test_dataset_length(sample_df, sample_config):
    dataset = TestDataset(sample_df, sample_config)
    expected_length = len(sample_df) - sample_config.time_series_window_in_hours - sample_config.forecasting_horizon_in_hours
    if expected_length <= 0:
        expected_length = 1
    assert len(dataset) == expected_length

def test_dataset_getitem(sample_df, sample_config):
    dataset = TestDataset(sample_df, sample_config)
    input_data, target = dataset[0]

    # Check input shape
    expected_input_features = sample_config.time_series_window_in_hours
    if sample_config.include_time_information:
        expected_input_features += 4  # Additional time features

    assert input_data.shape == torch.Size([expected_input_features])
    assert target.shape == torch.Size([1 if sample_config.is_single_time_point_prediction
                                       else sample_config.forecasting_horizon_in_hours])
    assert input_data.dtype == torch.float32
    assert target.dtype == torch.float32

def test_get_number_of_input_features(sample_df, sample_config):
    dataset = TestDataset(sample_df, sample_config)
    expected_features = sample_config.time_series_window_in_hours
    if sample_config.include_time_information:
        expected_features += 4
    assert dataset.get_number_of_input_features() == expected_features

def test_get_number_of_target_variables(sample_df, sample_config):
    dataset = TestDataset(sample_df, sample_config)
    expected_targets = 1 if sample_config.is_single_time_point_prediction else sample_config.forecasting_horizon_in_hours
    assert dataset.get_number_of_target_variables() == expected_targets

def test_dataset_with_minimal_data():
    # Test with minimal data (edge case)
    mini_config = DatasetConfig(
        time_variable='timestamp',
        target_variable='value',
        time_series_window_in_hours=2,
        forecasting_horizon_in_hours=1,
        is_single_time_point_prediction=True,
        include_time_information=False,
        time_series_scaler=None,
        is_training_set=True,
        labels_count=0,
        one_hot_time_variables=False
    )
    mini_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=3, freq='H'),
        'value': [1.0, 2.0, 3.0]
    })
    dataset = TestDataset(mini_df, mini_config)
    assert len(dataset) == 1
    assert dataset.get_number_of_input_features() == 2
    assert dataset.get_number_of_target_variables() == 1