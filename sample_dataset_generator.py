import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_dataset(
        start_date: str = "2015-01-01",
        end_date: str = "2016-12-31",
        interval_minutes: int = 15
) -> pd.DataFrame:
    """
    Create a sample time series dataset with the correct format.

    Args:
        start_date: Start date for the dataset
        end_date: End date for the dataset
        interval_minutes: Time interval between readings

    Returns:
        DataFrame with the required format
    """
    # Create date range
    date_range = pd.date_range(
        start=start_date,
        end=end_date,
        freq=f"{interval_minutes}min"
    )

    # Create base dataframe
    df = pd.DataFrame(index=date_range)

    # Add timestamp column
    df['timestamp'] = df.index.strftime('%Y-%m-%d %H:%M:%S')

    # Generate synthetic target values with daily and weekly patterns
    hours = df.index.hour + df.index.minute/60
    days = (df.index - df.index[0]).days

    # Daily pattern
    daily_pattern = 40 + 20 * np.sin(2 * np.pi * hours / 24)
    # Weekly pattern
    weekly_pattern = 10 * np.sin(2 * np.pi * days / 7)
    # Long term trend
    trend = days * 0.01
    # Random noise
    noise = np.random.normal(0, 2, len(df))

    # Combine patterns
    df['target'] = daily_pattern + weekly_pattern + trend + noise

    # Add time-based features
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

    # Add environmental features (synthetic)
    df['temperature'] = 20 + 5 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 1, len(df))
    df['humidity'] = 60 + 10 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 3, len(df))

    # Add special event flags
    df['is_holiday'] = (df.index.dayofweek.isin([5, 6]) |
                        df.index.strftime('%m-%d').isin(['01-01', '12-25'])).astype(int)

    # Ensure columns are in a sensible order
    column_order = [
        'timestamp', 'target', 'temperature', 'humidity',
        'hour_of_day', 'day_of_week', 'month', 'is_weekend', 'is_holiday'
    ]

    return df[column_order]

if __name__ == "__main__":
    # Create sample dataset
    df = create_sample_dataset()

    # Save to CSV
    df.to_csv('data/your_data.csv', index=False)

    # Print information about the dataset
    print("\nDataset Information:")
    print(f"Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"Number of rows: {len(df):,}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.1f} MB")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)