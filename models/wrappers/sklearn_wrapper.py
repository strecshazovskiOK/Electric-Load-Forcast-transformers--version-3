# models/wrappers/sklearn_wrapper.py
from typing import Optional, Dict, Any, Tuple, Union, Protocol
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from models.base.base_wrapper import BaseWrapper
from models.registry.model_types import ModelType
from training.reports.training_report import TrainingReport


class SklearnEstimator(Protocol):
    """Protocol defining the required interface for sklearn estimators."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnEstimator": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class SklearnWrapper(BaseWrapper):
    """Wrapper for scikit-learn models providing consistent interface."""

    def __init__(
            self,
            model: SklearnEstimator,  # Updated type hint using the Protocol
            model_type: ModelType,
            config: Dict[str, Any]
    ):
        super().__init__(model_type, config)
        self.model = model

        # Optional preprocessing
        self.scaler_x = StandardScaler() if config.get('scale_features', True) else None
        self.scaler_y = StandardScaler() if config.get('scale_target', True) else None

        # Track if the model has been fitted
        self.is_fitted = False

    def _prepare_data(
            self,
            dataset: Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert PyTorch dataset to numpy arrays."""
        X, y = [], []

        for features, target in dataset:
            X.append(features.numpy())
            y.append(target.numpy())

        return np.array(X), np.array(y)

    def _scale_data(
            self,
            X: np.ndarray,
            y: np.ndarray,
            fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features and target if scalers are enabled."""
        if self.scaler_x is not None:
            X = self.scaler_x.fit_transform(X) if fit else self.scaler_x.transform(X)

        if self.scaler_y is not None:
            y = self.scaler_y.fit_transform(y.reshape(-1, 1)) if fit else self.scaler_y.transform(y.reshape(-1, 1))
            y = y.ravel()

        return X, y

    def train(
            self,
            train_dataset: Dataset,
            validation_dataset: Optional[Dataset] = None
    ) -> TrainingReport:
        """Train the scikit-learn model."""
        # Prepare data
        X_train, y_train = self._prepare_data(train_dataset)

        # Scale data if needed
        X_train, y_train = self._scale_data(X_train, y_train, fit=True)

        # Fit model
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Create training report with initial empty lists
        report = TrainingReport(train_losses=[], val_losses=[])
        
        # Add the training score - convert numpy float to Python float
        train_mse = float(np.mean((self.model.predict(X_train) - y_train) ** 2))
        report.train_losses.append(train_mse)

        if validation_dataset:
            X_val, y_val = self._prepare_data(validation_dataset)
            X_val, y_val = self._scale_data(X_val, y_val, fit=False)
            val_mse = float(np.mean((self.model.predict(X_val) - y_val) ** 2))
            report.val_losses.append(val_mse)

        return report

    def predict(self, dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions using the scikit-learn model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")

        # Prepare data
        X, y = self._prepare_data(dataset)
        X, _ = self._scale_data(X, y, fit=False)

        # Make predictions
        y_pred = self.model.predict(X)

        # Inverse transform predictions if needed
        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        return torch.tensor(y_pred), torch.tensor(y)

    def save(self, path: str) -> None:
        """Save model state."""
        import joblib
        save_dict = {
            'model': self.model,
            'scaler_x': self.scaler_x,
            'scaler_y': self.scaler_y,
            'is_fitted': self.is_fitted,
            'config': self.config
        }
        joblib.dump(save_dict, path)

    def load(self, path: str) -> None:
        """Load model state."""
        import joblib
        load_dict = joblib.load(path)

        self.model = load_dict['model']
        self.scaler_x = load_dict['scaler_x']
        self.scaler_y = load_dict['scaler_y']
        self.is_fitted = load_dict['is_fitted']
        self.config.update(load_dict['config'])