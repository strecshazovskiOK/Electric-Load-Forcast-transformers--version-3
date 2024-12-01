# tests/unit/test_loaders.py
import pytest
import pandas as pd
from datetime import date
from pathlib import Path

from data_loading.loaders.time_series_loader import TimeInterval, TimeSeriesLoader


@pytest.fixture
def sample_csv(tmp_path):
    # Create a temporary CSV file
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'value': range(100)
    })
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def sample_intervals():
    return {
        'train': TimeInterval(date(2024, 1, 1), date(2024, 1, 2)),
        'val': TimeInterval(date(2024, 1, 3), date(2024, 1, 3)),
        'test': TimeInterval(date(2024, 1, 4), date(2024, 1, 4))
    }

def test_time_interval_overlap():
    interval1 = TimeInterval(date(2024, 1, 1), date(2024, 1, 3))
    interval2 = TimeInterval(date(2024, 1, 2), date(2024, 1, 4))
    interval3 = TimeInterval(date(2024, 1, 4), date(2024, 1, 5))

    assert interval1.is_interval_overlapping(interval2)
    assert not interval1.is_interval_overlapping(interval3)

def test_loader_initialization():
    loader = TimeSeriesLoader('timestamp', 'value')
    assert loader.time_variable == 'timestamp'
    assert loader.target_variable == 'value'
    assert loader.csv_dataframe is None

def test_loader_load(sample_csv):
    loader = TimeSeriesLoader('timestamp', 'value')
    df = loader.load(sample_csv)

    assert isinstance(df, pd.DataFrame)
    assert 'timestamp' in df.columns
    assert 'value' in df.columns
    assert isinstance(df['timestamp'].iloc[0], pd.Timestamp)

def test_loader_split(sample_csv, sample_intervals):
    loader = TimeSeriesLoader('timestamp', 'value')
    df = loader.load(sample_csv)

    train_data, val_data, test_data = loader.split(
        df,
        sample_intervals['train'],
        sample_intervals['val'],
        sample_intervals['test']
    )

    assert isinstance(train_data, pd.DataFrame)
    assert isinstance(val_data, pd.DataFrame)
    assert isinstance(test_data, pd.DataFrame)

    # Check that data is properly split
    assert all(train_data['timestamp'].dt.date <= sample_intervals['train'].max_date)
    assert all(val_data['timestamp'].dt.date <= sample_intervals['val'].max_date)
    assert all(test_data['timestamp'].dt.date <= sample_intervals['test'].max_date)

def test_overlapping_intervals_error(sample_csv):
    loader = TimeSeriesLoader('timestamp', 'value')
    df = loader.load(sample_csv)

    overlapping_intervals = {
        'train': TimeInterval(date(2024, 1, 1), date(2024, 1, 3)),
        'val': TimeInterval(date(2024, 1, 2), date(2024, 1, 4)),
        'test': TimeInterval(date(2024, 1, 4), date(2024, 1, 5))
    }

    with pytest.raises(ValueError):
        loader.split(
            df,
            overlapping_intervals['train'],
            overlapping_intervals['val'],
            overlapping_intervals['test']
        )

def test_empty_dataframe_error(sample_intervals):
    loader = TimeSeriesLoader('timestamp', 'value')
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError):
        loader.split(
            empty_df,
            sample_intervals['train'],
            sample_intervals['val'],
            sample_intervals['test']
        )