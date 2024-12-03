# data_loading/base/interval_types.py
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any

class TimeInterval(Enum):
    """Supported time intervals for prediction."""
    FIFTEEN_MIN = 15
    HOURLY = 60
    DAILY = 1440  # minutes in a day
    MONTHLY = 43200  # minutes in a month (approx)

    def get_points_per_hour(self) -> int:
        """Get number of data points per hour for this interval."""
        return 60 // self.value if self.value <= 60 else 1//(self.value // 60)
    
    def get_points_per_day(self) -> int:
        """Get number of data points per day for this interval."""
        return 1440 // self.value
        
    def get_points_per_month(self) -> int:
        """Get number of data points per month for this interval."""
        return 43200 // self.value

@dataclass
class IntervalConfig:
    """Configuration specific to time intervals."""
    interval_type: TimeInterval
    lookback_periods: int  # Number of periods to look back
    forecast_periods: int  # Number of periods to forecast
    
    def get_window_size(self) -> int:
        """Get total window size in data points."""
        if self.interval_type in {TimeInterval.FIFTEEN_MIN, TimeInterval.HOURLY}:
            return self.lookback_periods * self.interval_type.get_points_per_hour()
        elif self.interval_type == TimeInterval.DAILY:
            return self.lookback_periods
        else:  # MONTHLY
            return self.lookback_periods

    def get_horizon_size(self) -> int:
        """Get forecast horizon in data points."""
        if self.interval_type in {TimeInterval.FIFTEEN_MIN, TimeInterval.HOURLY}:
            return self.forecast_periods * self.interval_type.get_points_per_hour()
        elif self.interval_type == TimeInterval.DAILY:
            return self.forecast_periods
        else:  # MONTHLY
            return self.forecast_periods