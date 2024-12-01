# utils/losses/metrics.py
import torch
from torch import Tensor
from typing import Optional

from .base import CustomLoss

class MAPE(CustomLoss):
    """Mean Absolute Percentage Error loss."""

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Calculate MAPE loss.

        Args:
            output: Predicted values tensor
            target: Target values tensor

        Returns:
            MAPE loss value
        """
        error = torch.abs((target - output) / (torch.abs(target) + self.eps))
        return self._reduce(error) * 100

class MAE(CustomLoss):
    """Mean Absolute Error (L1) loss."""

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Calculate MAE loss.

        Args:
            output: Predicted values tensor
            target: Target values tensor

        Returns:
            MAE loss value
        """
        return self._reduce(torch.abs(target - output))

class NaiveForecast(CustomLoss):
    """Naive Forecast loss using seasonal patterns."""

    def __init__(
            self,
            seasonal_cycle: int,
            reduction: str = 'mean',
            eps: float = 1e-8
    ):
        """
        Args:
            seasonal_cycle: Number of steps after which seasonal pattern repeats
            reduction: Specifies the reduction to apply
            eps: Small constant for numerical stability
        """
        super().__init__(reduction, eps)
        self.seasonal_cycle = seasonal_cycle

    def forward(self, time_series: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """
        Calculate Naive Forecast loss.

        Args:
            time_series: Time series values tensor
            target: Unused, kept for interface consistency

        Returns:
            Naive forecast loss value
        """
        if time_series.dim() == 1:
            time_series = time_series.unsqueeze(0)

        errors = []
        for seq in time_series:
            seasonal_errors = torch.abs(
                seq[self.seasonal_cycle:] - seq[:-self.seasonal_cycle]
            )
            errors.append(self._reduce(seasonal_errors))

        return self._reduce(torch.stack(errors))

class MASE(CustomLoss):
    """Mean Absolute Scaled Error loss."""

    def __init__(
            self,
            seasonal_cycle: int,
            reduction: str = 'mean',
            eps: float = 1e-8
    ):
        """
        Args:
            seasonal_cycle: Number of steps after which seasonal pattern repeats
            reduction: Specifies the reduction to apply
            eps: Small constant for numerical stability
        """
        super().__init__(reduction, eps)
        self.seasonal_cycle = seasonal_cycle
        self.naive_forecast = NaiveForecast(seasonal_cycle, reduction, eps)
        self.mae = MAE(reduction, eps)

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Calculate MASE loss.

        Args:
            output: Predicted values tensor
            target: Target values tensor

        Returns:
            MASE loss value
        """
        mae_loss = self.mae(output, target)
        naive_loss = self.naive_forecast(target)
        return mae_loss / (naive_loss + self.eps)