import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional

class DataScaler:
    """Handles data scaling operations"""

    def __init__(self, scaler: Optional[StandardScaler] = None):
        self.scaler = scaler or StandardScaler()

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit scaler to data and transform"""
        print(f"\nDebug - DataScaler fit_transform:")
        print(f"Input data range: [{np.min(data):.2f}, {np.max(data):.2f}]")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        scaled_data = self.scaler.fit_transform(data).flatten()
        print(f"Output scaled range: [{np.min(scaled_data):.2f}, {np.max(scaled_data):.2f}]")
        print(f"Mean: {np.mean(scaled_data):.2f}, Std: {np.std(scaled_data):.2f}")
        
        return scaled_data

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler"""
        print(f"\nDebug - DataScaler transform:")
        print(f"Input data range: [{np.min(data):.2f}, {np.max(data):.2f}]")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        scaled_data = self.scaler.transform(data).flatten()
        print(f"Output scaled range: [{np.min(scaled_data):.2f}, {np.max(scaled_data):.2f}]")
        print(f"Mean: {np.mean(scaled_data):.2f}, Std: {np.std(scaled_data):.2f}")
        
        return scaled_data