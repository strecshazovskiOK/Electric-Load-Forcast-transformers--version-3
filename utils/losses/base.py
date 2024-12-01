# utils/losses/base.py
from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
from typing import Optional

class CustomLoss(nn.Module, ABC):
    """Base class for all custom loss functions."""

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        """
        Args:
            reduction: Specifies the reduction to apply to the output.
                    Options: ['none', 'mean', 'sum']
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    @abstractmethod
    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """Forward pass for the loss calculation."""
        pass

    def _reduce(self, loss: Tensor) -> Tensor:
        """Apply reduction to the loss."""
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")