import numpy as np
import pandas as pd

class DataTransformer:
    """Handles data transformation operations"""

    @staticmethod
    def extract_timestamps(df: pd.DataFrame, time_col: str, step: int = 1) -> np.ndarray:
        """Extract timestamps at regular intervals. Default step is 1 since data is already in 15-min intervals"""
        timestamps = np.array(df[time_col])
        print(f"\nDebug - Extracted timestamps:")
        print(f"First timestamp: {timestamps[0]}")
        print(f"Last timestamp: {timestamps[-1]}")
        return timestamps