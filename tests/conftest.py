import itertools
import pytest
import torch
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch.utils.data

from data_loading.base.base_dataset import DatasetConfig



# Model-related fixtures
@pytest.fixture
def device():
    """Provide the appropriate torch device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def batch_size():
    """Standard batch size for testing."""
    return 32

@pytest.fixture
def input_dim():
    """Standard input dimension for testing."""
    return 10

@pytest.fixture
def seq_length():
    """Standard sequence length for testing."""
    return 24

# Data generation fixtures
@pytest.fixture
def sample_timeseries_data(tmp_path):
    """Generate sample time series data with regular intervals."""
    dates = []
    values = []
    start_date = pd.Timestamp('2024-01-01')

    # Create realistic patterns
    base_load = 1000
    # Generate 4 measurements per hour for 100 hours
    for hour, quarter in itertools.product(range(100), range(4)):
        timestamp = start_date + pd.Timedelta(hours=hour) + pd.Timedelta(minutes=15*quarter)

        hour_of_day = timestamp.hour
        day_of_week = timestamp.dayofweek

        # Daily pattern
        hour_pattern = 200 * np.sin(2 * np.pi * (hour_of_day + quarter/4) / 24)
        # Weekly pattern
        week_pattern = -100 if day_of_week >= 5 else 100
        # Random noise
        noise = np.random.normal(0, 20)

        dates.append(timestamp)
        values.append(base_load + hour_pattern + week_pattern + noise)

    df = pd.DataFrame({
        'timestamp': dates,
        'value': values
    })

    # Save to CSV
    csv_path = tmp_path / "sample_timeseries.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def irregular_timeseries_data(tmp_path):
    """Generate irregular time series data with varying intervals."""
    dates = []
    values = []
    start_date = pd.Timestamp('2024-01-01')
    
    for hour in range(100):
        # Random number of measurements per hour (1-6)
        n_measurements = np.random.randint(1, 7)
        for _ in range(n_measurements):
            # Random minute within the hour
            timestamp = start_date + pd.Timedelta(hours=hour) + pd.Timedelta(minutes=np.random.randint(0, 60))
            value = 1000 + 200 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 20)
            dates.append(timestamp)
            values.append(value)
    
    df = pd.DataFrame({
        'timestamp': sorted(dates),  # Sort by timestamp
        'value': values
    })
    csv_path = tmp_path / "irregular_timeseries.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def large_timeseries_data(tmp_path):
    """Generate a large dataset for stress testing (1 year of data)."""
    dates = pd.date_range(start='2024-01-01', periods=365*24, freq='H')  # Hourly data for 1 year
    values = []
    
    for timestamp in dates:
        hour = timestamp.hour
        day = timestamp.dayofweek
        month = timestamp.month
        
        # Complex seasonal patterns
        hourly_pattern = 1000 + 200 * np.sin(2 * np.pi * hour / 24)
        weekly_pattern = 100 * np.sin(2 * np.pi * day / 7)
        yearly_pattern = 300 * np.sin(2 * np.pi * (month - 1) / 12)
        noise = np.random.normal(0, 50)
        
        values.append(hourly_pattern + weekly_pattern + yearly_pattern + noise)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'value': values
    })
    csv_path = tmp_path / "large_timeseries.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

# Configuration fixtures
@pytest.fixture
def base_config():
    """Basic configuration for testing."""
    return {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,
        'early_stopping': True,
        'patience': 3
    }

@pytest.fixture
def time_intervals():
    """Standard time intervals for train/val/test splits."""
    return {
        'train': (date(2024, 1, 1), date(2024, 1, 20)),
        'val': (date(2024, 1, 21), date(2024, 1, 25)),
        'test': (date(2024, 1, 26), date(2024, 1, 31))
    }

@pytest.fixture
def sample_model_config():
    """Basic model configuration for testing."""
    return {
        'input_features': 10,
        'hidden_dims': [64, 32],
        'output_dim': 1,
        'dropout': 0.1
    }

# Preprocessing fixtures
@pytest.fixture
def standard_scaler():
    """Provide a standard scaler instance."""
    return StandardScaler()

@pytest.fixture
def sample_dataloader():
    """Create a sample PyTorch DataLoader."""
    # Create random data
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Utility fixtures
@pytest.fixture
def temp_model_path(tmp_path):
    """Provide a temporary path for saving models."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    return model_dir

@pytest.fixture
def dataset_config():
    """Create dataset configuration for testing."""
    from sklearn.preprocessing import StandardScaler
    
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
def sample_data_csv(tmp_path):
    """Create a sample CSV file with time series data."""
    # Use your existing sample_timeseries_data implementation
    return sample_timeseries_data(tmp_path)