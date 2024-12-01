# data_loading/features/time_features.py
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Union
from workalendar.europe import BadenWurttemberg
from .base_feature import BaseFeature

def convert_to_datetime(dt: Union[datetime, np.datetime64, pd.Timestamp]) -> datetime:
    """Helper function to safely convert various datetime types to Python datetime"""
    if isinstance(dt, np.datetime64):
        return pd.Timestamp(dt).to_pydatetime()
    elif isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime()
    return dt

class CyclicalTimeFeature(BaseFeature):
    """Generates cyclical time features (sin/cos encoding)"""

    def __init__(self, period_length: int):
        self.period_length = period_length

    def generate(self, time_stamps: Union[pd.Series, np.ndarray]) -> List[float]:
        """Generate cyclical time features from timestamps

        Args:
            time_stamps: Series or array of timestamps to process

        Returns:
            List of sine and cosine encodings for the time value
        """
        # Handle single value vs array/series
        if isinstance(time_stamps, (pd.Series, np.ndarray)):
            # If we received a single value wrapped in a Series/array
            if len(time_stamps) == 1:
                time_value = time_stamps.iloc[0] if isinstance(time_stamps, pd.Series) else time_stamps[0]
            else:
                raise ValueError("CyclicalTimeFeature expects a single timestamp value")
        else:
            time_value = time_stamps

        # Convert to integer value based on input type
        if isinstance(time_value, (pd.Timestamp, np.datetime64)):
            try:
                time_value = pd.Timestamp(time_value)
                if hasattr(time_value, 'hour'):
                    numeric_value = time_value.hour
                elif hasattr(time_value, 'weekofyear'):
                    numeric_value = time_value.weekofyear
                else:
                    raise AttributeError(f"Unsupported time attribute for value: {time_value}")
            except Exception as e:
                print(f"Debug: Error processing time value: {time_value}, type: {type(time_value)}")
                raise
        else:
            numeric_value = int(time_value)

        # Generate cyclical features
        return [
            math.sin(2 * math.pi * numeric_value / self.period_length),
            math.cos(2 * math.pi * numeric_value / self.period_length)
        ]

    def get_feature_dim(self) -> int:
        """Return the dimension of the generated feature (sin and cos components)"""
        return 2

class OneHotTimeFeature(BaseFeature):
    """Generates one-hot encoded time features"""

    def __init__(self, period_length: int):
        self.period_length = period_length

    def generate(self, time_value: int) -> List[float]:
        encoding = np.zeros(self.period_length, dtype=float)
        encoding[time_value] = 1.0
        return encoding.tolist()

    def get_feature_dim(self) -> int:
        return self.period_length

class HourOfWeekFeature(BaseFeature):
    """Generates hour of week features"""
    DAY_IN_HOURS = 24

    def generate(self, dt: Union[datetime, np.datetime64, pd.Timestamp]) -> List[float]:
        dt = convert_to_datetime(dt)
        hour_of_week = float(dt.weekday() * self.DAY_IN_HOURS + dt.hour)
        return [hour_of_week]

    def get_feature_dim(self) -> int:
        return 1

class WorkdayFeature(BaseFeature):
    """Generates workday-related features"""

    def __init__(self):
        self.calendar = BadenWurttemberg()

    def generate(self, time_stamp: Union[datetime, np.datetime64, pd.Timestamp]) -> List[float]:
        time_stamp = convert_to_datetime(time_stamp)
        
        # Get previous and next days
        prev_day = time_stamp - timedelta(days=1)
        next_day = time_stamp + timedelta(days=1)
        
        is_workday = float(self.calendar.is_working_day(time_stamp))
        is_holiday = float(self.calendar.is_holiday(time_stamp))
        is_prev_workday = float(self.calendar.is_working_day(prev_day))
        is_next_workday = float(self.calendar.is_working_day(next_day))
        
        # Christmas period check
        christmas_start = datetime(time_stamp.year, 12, 23)
        christmas_end = datetime(time_stamp.year, 12, 28)
        is_christmas = float(christmas_start.date() < time_stamp.date() < christmas_end.date())
        
        return [is_workday, is_holiday, is_prev_workday, is_next_workday, is_christmas]

    def get_feature_dim(self) -> int:
        return 5