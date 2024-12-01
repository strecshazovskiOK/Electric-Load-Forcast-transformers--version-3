# data_loading/features/base_feature.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Union

class BaseFeature(ABC):
    """Base class for all feature generators"""

    @abstractmethod
    def generate(self, time_stamps: Union[pd.Series, np.ndarray]) -> List[float]:
        """Generate features from timestamps"""
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the dimension of the generated feature"""
        pass