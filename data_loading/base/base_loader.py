# data_loading/base/base_loader.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union
import pandas as pd

class BaseLoader(ABC):
    """Base class for all data loaders"""

    @abstractmethod
    def load(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load data from source"""
        pass

    @abstractmethod
    def split(
            self,
            df: pd.DataFrame,
            train_interval,
            validation_interval,
            test_interval
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation and test sets"""
        pass
