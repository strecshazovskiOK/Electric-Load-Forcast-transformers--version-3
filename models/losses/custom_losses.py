# models/losses/custom_losses.py
import torch
from torch import nn, Tensor
from typing import Dict, List

class MAPE(nn.Module):
    """Mean Absolute Percentage Error loss."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Calculate MAPE loss.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            MAPE loss value
        """
        return torch.mean(
            torch.abs(
                (target - pred) / (torch.abs(target) + self.eps)
            )
        ) * 100

class RobustMSELoss(nn.Module):
    """MSE loss with outlier handling."""
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute Huber-like loss that's more robust to outliers.
        """
        diff = target - pred
        loss = torch.where(
            torch.abs(diff) < self.beta,
            0.5 * diff ** 2,
            self.beta * torch.abs(diff) - 0.5 * self.beta ** 2
        )
        return torch.mean(loss)

class CombinedLoss(nn.Module):
    """Combination of multiple loss functions with weights."""

    def __init__(self, losses: Dict[nn.Module, float]):
        """
        Args:
            losses: Dictionary mapping loss functions to their weights
        """
        super().__init__()
        self.losses = losses

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Calculate combined loss.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            Weighted sum of all losses
        """
        total_loss = 0.0
        for loss_fn, weight in self.losses.items():
            total_loss += weight * loss_fn(pred, target)
        return torch.tensor(total_loss, device=pred.device)