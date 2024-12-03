from typing import List, Dict, Any, Optional
import numpy as np

class TrainingReport:
    def __init__(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        learning_rates: Optional[List[float]] = None,
        epochs: Optional[int] = None,
        early_stopping_epoch: Optional[int] = None,
        additional_metrics: Optional[Dict[str, List[float]]] = None,
        metrics: Optional[Dict[str, Any]] = None  # Add for backward compatibility
    ):
        self.train_losses = train_losses
        self.val_losses = val_losses if val_losses is not None else []
        self.learning_rates = learning_rates if learning_rates is not None else []
        self.epochs = epochs
        self.early_stopping_epoch = early_stopping_epoch
        self.additional_metrics = additional_metrics if additional_metrics is not None else {}
        if metrics:
            self.additional_metrics.update(metrics)

    def add_loss(self, train_loss: float, val_loss: Optional[float] = None, lr: Optional[float] = None) -> None:
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if lr is not None:
            self.learning_rates.append(lr)

    def add_metric(self, name: str, value: float) -> None:
        if name not in self.additional_metrics:
            self.additional_metrics[name] = []
        self.additional_metrics[name].append(value)

    def get_best_epoch(self, monitor: str = 'val_loss') -> int:
        if monitor == 'val_loss' and self.val_losses:
            return int(np.argmin(self.val_losses))
        return int(np.argmin(self.train_losses))

    def serialize(self) -> Dict[str, Any]:
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'epochs': self.epochs,
            'early_stopping_epoch': self.early_stopping_epoch,
            'additional_metrics': self.additional_metrics
        }

    @staticmethod
    def deserialize(data: Optional[Dict]) -> Optional['TrainingReport']:
        if data is None:
            return None
        return TrainingReport(
            train_losses=data['train_losses'],
            val_losses=data.get('val_losses'),
            learning_rates=data.get('learning_rates'),
            epochs=data.get('epochs'),
            early_stopping_epoch=data.get('early_stopping_epoch'),
            additional_metrics=data.get('additional_metrics')
        )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional['TrainingReport']:
        """
        Creates a TrainingReport instance from dictionary data.
        Args:
            data: Dictionary containing training report data or None
        Returns:
            TrainingReport instance or None if data is None
        """
        if data is None:
            return None
            
        return cls(
            train_losses=data['train_losses'],
            val_losses=data.get('val_losses'),
            learning_rates=data.get('learning_rates'),
            epochs=data.get('epochs'),
            early_stopping_epoch=data.get('early_stopping_epoch'),
            additional_metrics=data.get('additional_metrics')
        )

    def __str__(self) -> str:
        return (
            f"Training Report:\n"
            f"Epochs: {self.epochs}\n"
            f"Final train loss: {self.train_losses[-1] if self.train_losses else 'N/A'}\n"
            f"Final val loss: {self.val_losses[-1] if self.val_losses else 'N/A'}\n"
            f"Early stopping epoch: {self.early_stopping_epoch if self.early_stopping_epoch else 'N/A'}"
        )