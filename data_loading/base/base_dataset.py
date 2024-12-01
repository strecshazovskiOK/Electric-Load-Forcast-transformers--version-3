# path: data_loading/base/base_dataset.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Any, Optional, Dict
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""
    # Required parameters
    time_variable: str
    target_variable: str
    time_series_window_in_hours: int
    forecasting_horizon_in_hours: int
    is_single_time_point_prediction: bool
    include_time_information: bool
    is_training_set: bool
    labels_count: int
    one_hot_time_variables: bool

    # Data processing parameters with defaults
    normalize_data: bool = False
    scaling_method: str = "standard"
    time_series_scaler: Any = None

    # Time resolution configuration
    time_resolution_minutes: int = 15
    points_per_hour: int = field(init=False)

    # Feature generation flags
    add_time_features: bool = True
    add_holiday_features: bool = False
    add_weather_features: bool = False

    # Cache for computed values
    _window_size: Optional[int] = field(init=False, default=None)
    _horizon_size: Optional[int] = field(init=False, default=None)

    def __post_init__(self):
        """Validate configuration and compute derived values."""
        # Validate time resolution
        if 60 % self.time_resolution_minutes != 0:
            raise ValueError(f"Invalid time resolution: {self.time_resolution_minutes} minutes")
        
        # Set points per hour
        self.points_per_hour = 60 // self.time_resolution_minutes

        # Initialize scaler if needed
        if self.normalize_data and self.time_series_scaler is None:
            self.time_series_scaler = StandardScaler() if self.scaling_method == "standard" else None

        # Compute and cache window sizes in data points
        self._window_size = self.time_series_window_in_hours * self.points_per_hour
        self._horizon_size = self.forecasting_horizon_in_hours * self.points_per_hour

    @property
    def window_size(self) -> int:
        """Get window size in number of data points."""
        return self._window_size if self._window_size is not None else self.time_series_window_in_hours * self.points_per_hour

    @property
    def horizon_size(self) -> int:
        """Get forecast horizon in number of data points."""
        return self._horizon_size if self._horizon_size is not None else self.forecasting_horizon_in_hours * self.points_per_hour

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'time_variable': self.time_variable,
            'target_variable': self.target_variable,
            'time_series_window_in_hours': self.time_series_window_in_hours,
            'forecasting_horizon_in_hours': self.forecasting_horizon_in_hours,
            'is_single_time_point_prediction': self.is_single_time_point_prediction,
            'include_time_information': self.include_time_information,
            'is_training_set': self.is_training_set,
            'labels_count': self.labels_count,
            'one_hot_time_variables': self.one_hot_time_variables,
            'normalize_data': self.normalize_data,
            'scaling_method': self.scaling_method,
            'time_resolution_minutes': self.time_resolution_minutes,
            'add_time_features': self.add_time_features,
            'add_holiday_features': self.add_holiday_features,
            'add_weather_features': self.add_weather_features
        }


class BaseDataset(ABC, Dataset):
    """Base class for all datasets"""

    def __init__(self, df: pd.DataFrame, config: DatasetConfig):
        self._df = df
        self.config = config
        self.prepared_time_series_input: Optional[torch.Tensor] = None
        self.prepared_time_series_target: Optional[torch.Tensor] = None
        self.time_labels: Optional[np.ndarray] = None
        self.rows: Optional[torch.Tensor] = None

    @abstractmethod
    def _prepare_time_series_data(self) -> None:
        """Prepare time series data for model input"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the dataset"""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get a data item by index"""
        pass

    def get_number_of_input_features(self) -> int:
        """Return number of input features"""
        if hasattr(self, 'prepared_time_series_input') and self.prepared_time_series_input is not None:
            return self.prepared_time_series_input.shape[1]
        return 0

    def get_number_of_target_variables(self) -> int:
        """Return number of target variables"""
        if hasattr(self, 'prepared_time_series_target') and self.prepared_time_series_target is not None:
            return self.prepared_time_series_target.shape[1]
        return 0

    def _validate_data(self) -> None:
        """Validate input data format and completeness."""
        if self._df is None or self._df.empty:
            raise ValueError("Input DataFrame is empty or None")

        required_columns = [self.config.time_variable, self.config.target_variable]
        missing_columns = [col for col in required_columns if col not in self._df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")