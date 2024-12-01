# data_loading/loaders/time_series_loader.py
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Tuple, Union, cast
import pandas as pd
from pandas import DataFrame

from data_loading.base.base_loader import BaseLoader


@dataclass
class TimeInterval:
    """Represents a time interval with inclusive bounds"""
    min_date: date
    max_date: date  # Fixed: Added type annotation

    def is_interval_overlapping(self, other: 'TimeInterval') -> bool:
        """Check if this interval overlaps with another interval"""
        return not (
            self.min_date > other.max_date or
            self.max_date < other.min_date
        )

class TimeSeriesLoader(BaseLoader):
    """Loads and splits time series data from CSV files"""

    def __init__(self, time_variable: str, target_variable: str):
        self.time_variable = time_variable
        self.target_variable = target_variable
        self.csv_dataframe = None

    def _load_dataframe_from_csv(self, path: Union[str, Path], columns_to_parse_as_dates: List[str], columns_to_include: List[str]) -> pd.DataFrame:
        """Load CSV file into DataFrame with specific parsing requirements"""
        try:
            df = pd.read_csv(
                path,
                usecols=columns_to_include,
                parse_dates=columns_to_parse_as_dates
            )
            
            if missing_cols := set(columns_to_include) - set(df.columns):  # Using walrus operator
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            return df
            
        except Exception as e:
            raise IOError(f"Error loading CSV file: {str(e)}") from e

    def load(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load data from CSV file"""
        self.csv_dataframe = self._load_dataframe_from_csv(
            path=path,
            columns_to_parse_as_dates=[self.time_variable],
            columns_to_include=[self.time_variable, self.target_variable]
        )
        print("Debug: Loaded DataFrame time column type:", self.csv_dataframe[self.time_variable].dtype)
        sample_time = pd.Timestamp(cast(pd.Timestamp, self.csv_dataframe[self.time_variable].iloc[0])) # type: ignore
        print("Debug: Sample time value:", sample_time)
        return self.csv_dataframe

    def split(
            self,
            df: pd.DataFrame,
            train_interval: TimeInterval,
            validation_interval: TimeInterval,
            test_interval: TimeInterval
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation and test sets"""
        print("Debug: Splitting data...")

        # Debug: Log intervals
        print("Debug: Train Interval:", train_interval)
        print("Debug: Validation Interval:", validation_interval)
        print("Debug: Test Interval:", test_interval)

        if df is None or df.empty:
            raise ValueError("Input DataFrame must not be None or empty")

        if (train_interval.is_interval_overlapping(validation_interval) or
                validation_interval.is_interval_overlapping(test_interval)):
            raise ValueError("Train, validation and test intervals must not overlap")

        train_data = self._extract_by_interval(df, train_interval)
        val_data = self._extract_by_interval(df, validation_interval)
        test_data = self._extract_by_interval(df, test_interval)

        # Debug: Log split results
        print("Debug: Train data shape:", train_data.shape)
        print("Debug: Validation data shape:", val_data.shape)
        print("Debug: Test data shape:", test_data.shape)

        return train_data, val_data, test_data

    def _extract_by_interval(self, df: pd.DataFrame, interval: TimeInterval) -> pd.DataFrame:
        """Extract subset of data within given time interval"""
        filtered_df = df[
            (df[self.time_variable].dt.date >= interval.min_date) &
            (df[self.time_variable].dt.date <= interval.max_date)
        ]
        
        if filtered_df.empty:
            print(f"Warning: No data found for interval {interval.min_date} to {interval.max_date}")
            # Create a minimal dataframe with the same columns but a single row of default values
            default_row = pd.DataFrame({
                self.time_variable: [pd.Timestamp(interval.min_date)],
                self.target_variable: [0.0]
            })
            return default_row
        
        return filtered_df