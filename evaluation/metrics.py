# evaluation/metrics.py
from typing import Dict

import torch
import numpy as np
from dataclasses import dataclass

from torch import Tensor


@dataclass
class MetricConfig:
    """Configuration for metric calculation."""
    seasonal_period: int = 168  # Default weekly seasonality
    epsilon: float = 1e-8  # Small value for numerical stability

class Metrics:
    """Collection of evaluation metrics."""

    @staticmethod
    def mape(predicted: Tensor, expected: Tensor, epsilon: float = 1e-8) -> float:
        """Mean Absolute Percentage Error."""
        # Ensure tensors are properly aligned
        error = torch.abs((expected - predicted) / (torch.abs(expected) + epsilon))
        return torch.mean(error).item() * 100

    @staticmethod
    def mase(predicted: Tensor, expected: Tensor, seasonal_period: int) -> float:
        """Mean Absolute Scaled Error."""
        # Ensure inputs are 2D: [time, features]
        if predicted.dim() > 2:
            predicted = predicted.reshape(-1, predicted.size(-1))
            expected = expected.reshape(-1, expected.size(-1))

        # Calculate MAE
        mae = torch.mean(torch.abs(expected - predicted))

        # Calculate denominator (mean absolute error of naive seasonal forecast)
        naive_errors = []
        for i in range(seasonal_period, len(expected)):
            error = torch.abs(expected[i] - expected[i - seasonal_period])
            naive_errors.append(error)
        
        if not naive_errors:
            return float('nan')
            
        naive_errors_tensor = torch.stack(naive_errors)
        naive_mae = torch.mean(naive_errors_tensor)
        
        if naive_mae == 0:
            return float('inf')
            
        return (mae / naive_mae).item()

    @staticmethod
    def rmse(predicted: Tensor, expected: Tensor) -> float:
        """Root Mean Square Error."""
        # Ensure tensors are properly aligned
        return torch.sqrt(torch.mean((predicted - expected) ** 2)).item()

    @staticmethod
    def mae(predicted: Tensor, expected: Tensor) -> float:
        """Mean Absolute Error."""
        # Ensure tensors are properly aligned
        return torch.mean(torch.abs(expected - predicted)).item()

    @staticmethod
    def calculate_all_metrics(predicted: Tensor, expected: Tensor, config: MetricConfig) -> Dict[str, float]:
        """Calculate all available metrics."""
        try:
            mase_value = Metrics.mase(predicted, expected, config.seasonal_period)
        except:
            mase_value = float('nan')

        return {
            'mape': Metrics.mape(predicted, expected, config.epsilon),
            'mase': mase_value,
            'rmse': Metrics.rmse(predicted, expected),
            'mae': Metrics.mae(predicted, expected)
        }