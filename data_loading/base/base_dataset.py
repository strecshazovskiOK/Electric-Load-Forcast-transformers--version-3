# path: data_loading/base/base_dataset.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional, Dict
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_loading.types.interval_types import IntervalConfig, TimeInterval

@dataclass
class DatasetConfig:
    """Configuration for dataset preparation with enhanced feature control."""
    # Required core parameters
    time_variable: str
    target_variable: str
    
    # Time resolution parameters
    input_resolution_minutes: int
    forecast_resolution_minutes: int
    
    # Window and horizon configuration
    lookback_periods: int
    forecast_periods: int
    
    # Core feature and prediction configuration
    is_single_time_point_prediction: bool
    include_time_information: bool
    is_training_set: bool
    labels_count: int
    one_hot_time_variables: bool

    # Data processing parameters with defaults
    normalize_data: bool = False
    scaling_method: str = "standard"
    time_series_scaler: Any = None
    handle_missing_values: str = "interpolate"
    remove_outliers: bool = False
    outlier_std_threshold: float = 3.0

    # Basic feature generation flags
    add_time_features: bool = True
    add_holiday_features: bool = False
    add_weather_features: bool = False

    # Detailed time feature controls
    add_hour_feature: bool = True
    add_weekday_feature: bool = True
    add_month_feature: bool = True
    add_season_feature: bool = False
    add_year_feature: bool = False
    
    # Data augmentation settings
    use_data_augmentation: bool = False
    augmentation_methods: List[str] = field(default_factory=lambda: ["jitter", "scaling"])
    augmentation_probability: float = 0.3

    # Sequence handling
    padding_value: float = 0.0
    mask_padding: bool = True
    max_sequence_gaps: int = 3

    # Cache for computed values
    _window_size: Optional[int] = field(init=False, default=None)
    _horizon_size: Optional[int] = field(init=False, default=None)
    _interval_config: Optional[IntervalConfig] = field(init=False, default=None)
    _input_points_per_period: Optional[int] = field(init=False, default=None)
    _forecast_points_per_period: Optional[int] = field(init=False, default=None)

    def __post_init__(self):
        """Validate configuration and compute derived values."""
        self._validate_resolutions()
        self._initialize_interval_config()
        self._initialize_scaler()
        self._compute_points_per_period()
        self._compute_window_sizes()


    def _validate_resolutions(self) -> None:
        """Validate input and forecast resolutions."""
        # Validate input resolution
        if 60 % self.input_resolution_minutes != 0 and self.input_resolution_minutes % 60 != 0:
            raise ValueError(f"Invalid input resolution: {self.input_resolution_minutes} minutes")
            
        # Validate forecast resolution
        if 60 % self.forecast_resolution_minutes != 0 and self.forecast_resolution_minutes % 60 != 0:
            raise ValueError(f"Invalid forecast resolution: {self.forecast_resolution_minutes} minutes")
            
        # Validate that forecast resolution is not finer than input resolution
        if self.forecast_resolution_minutes < self.input_resolution_minutes:
            raise ValueError(
                f"Forecast resolution ({self.forecast_resolution_minutes} min) cannot be finer than "
                f"input resolution ({self.input_resolution_minutes} min)"
            )

    def _initialize_interval_config(self) -> None:
        """Initialize the interval configuration."""
        self._interval_config = IntervalConfig(
            interval_type=self._determine_interval_type(),
            lookback_periods=self.lookback_periods,
            forecast_periods=self.forecast_periods
        )

    def _initialize_scaler(self) -> None:
        """Initialize the data scaler if needed."""
        if self.normalize_data and self.time_series_scaler is None:
            self.time_series_scaler = StandardScaler() if self.scaling_method == "standard" else None

    def _compute_points_per_period(self) -> None:
        """Compute points per period for both input and forecast resolutions."""
        self._input_points_per_period = (
            60 // self.input_resolution_minutes 
            if self.input_resolution_minutes <= 60 
            else 1
        )
        self._forecast_points_per_period = (
            60 // self.forecast_resolution_minutes 
            if self.forecast_resolution_minutes <= 60 
            else 1
        )

    def _compute_window_sizes(self) -> None:
        """Compute window sizes for input and forecast windows."""
        if self._input_points_per_period is None or self._forecast_points_per_period is None:
            self._compute_points_per_period()
        
        input_points = self._input_points_per_period or 0  # Default to 0 if None
        forecast_points = self._forecast_points_per_period or 0  # Default to 0 if None
            
        self._window_size = self.lookback_periods * input_points
        self._horizon_size = self.forecast_periods * forecast_points

    def _determine_interval_type(self) -> TimeInterval:
        """Determine the interval type based on forecast resolution."""
        resolution = self.forecast_resolution_minutes
        if resolution <= 15:
            return TimeInterval.FIFTEEN_MIN
        elif resolution <= 60:
            return TimeInterval.HOURLY
        elif resolution <= 1440:  # 24 hours
            return TimeInterval.DAILY
        else:
            return TimeInterval.MONTHLY

    @property
    def window_size(self) -> int:
        """Get window size in number of data points."""
        if self._window_size is None:
            self._compute_window_sizes()
        assert self._window_size is not None
        return self._window_size

    @property
    def horizon_size(self) -> int:
        """Get forecast horizon in number of data points."""
        if self._horizon_size is None:
            self._compute_window_sizes()
        assert self._horizon_size is not None
        return self._horizon_size

    @property
    def input_points_per_period(self) -> int:
        """Get number of input data points per period."""
        if self._input_points_per_period is None:
            self._compute_points_per_period()
        assert self._input_points_per_period is not None
        return self._input_points_per_period

    @property
    def forecast_points_per_period(self) -> int:
        """Get number of forecast points per period."""
        if self._forecast_points_per_period is None:
            self._compute_points_per_period()
        assert self._forecast_points_per_period is not None
        return self._forecast_points_per_period

    @property
    def time_series_window_in_hours(self) -> int:
        """Get lookback window size in hours."""
        return self.lookback_periods * (self.input_resolution_minutes // 60) \
            if self.input_resolution_minutes >= 60 \
            else self.lookback_periods // (60 // self.input_resolution_minutes)

    @property
    def forecasting_horizon_in_hours(self) -> int:
        """Get forecast horizon in hours."""
        return self.forecast_periods * (self.forecast_resolution_minutes // 60) \
            if self.forecast_resolution_minutes >= 60 \
            else self.forecast_periods // (60 // self.forecast_resolution_minutes)

    @property
    def points_per_hour(self) -> int:
        """Get number of data points per hour."""
        return 60 // self.input_resolution_minutes if self.input_resolution_minutes <= 60 else 1

    def needs_resampling(self) -> bool:
        """Check if input data needs resampling for forecasting."""
        return self.input_resolution_minutes != self.forecast_resolution_minutes

    def get_resampling_factor(self) -> float:
        """Get the factor by which to resample the data."""
        return self.forecast_resolution_minutes / self.input_resolution_minutes

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'time_variable': self.time_variable,
            'target_variable': self.target_variable,
            'input_resolution_minutes': self.input_resolution_minutes,
            'forecast_resolution_minutes': self.forecast_resolution_minutes,
            'lookback_periods': self.lookback_periods,
            'forecast_periods': self.forecast_periods,
            'is_single_time_point_prediction': self.is_single_time_point_prediction,
            'include_time_information': self.include_time_information,
            'is_training_set': self.is_training_set,
            'labels_count': self.labels_count,
            'one_hot_time_variables': self.one_hot_time_variables,
            'normalize_data': self.normalize_data,
            'scaling_method': self.scaling_method,
            'add_time_features': self.add_time_features,
            'add_holiday_features': self.add_holiday_features,
            'add_weather_features': self.add_weather_features
        }

    def get_interval_config(self) -> Optional[IntervalConfig]:
        """Get the interval configuration."""
        return self._interval_config



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
        if missing_columns := [
            col for col in required_columns if col not in self._df.columns
        ]:
            raise ValueError(f"Missing required columns: {missing_columns}")