# evaluation/evaluator.py
import contextlib
from typing import Optional, Tuple, List
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor
import numpy as np

from .metrics import Metrics, MetricConfig
from .results import EvaluationResult, PredictionComparison

class Evaluator:
    """Evaluates model predictions against actual values."""

    def __init__(
            self,
            scaler: Optional[StandardScaler] = None,
            metric_config: Optional[MetricConfig] = None
    ):
        self.scaler = scaler
        self.metric_config = metric_config or MetricConfig()

    def evaluate(
            self,
            predictions: Tensor,
            targets: Tensor,
            timestamps: np.ndarray,
            num_variables: int = 1
    ) -> EvaluationResult:
        """
        Evaluate model predictions.

        Args:
            predictions: Model predictions
            targets: Actual target values
            timestamps: Timestamps for each prediction
            num_variables: Number of target variables

        Returns:
            Evaluation results including metrics and comparisons
        """
        # Unscale predictions if scaler is provided
        pred_np, target_np = self._prepare_data(predictions, targets)

        # Calculate overall metrics
        total_metrics = Metrics.calculate_all_metrics(
            torch.tensor(pred_np),
            torch.tensor(target_np),
            self.metric_config
        )

        # Calculate per-variable metrics if multiple variables
        variable_metrics = {}
        if num_variables > 1:
            for i in range(num_variables):
                variable_metrics[f'var_{i}'] = Metrics.calculate_all_metrics(
                    torch.tensor(pred_np[:, i]),
                    torch.tensor(target_np[:, i]),
                    self.metric_config
                )

        # Create detailed comparisons
        comparisons = self._create_comparisons(pred_np, target_np, timestamps)

        return EvaluationResult(
            total_metrics=total_metrics,
            variable_metrics=variable_metrics,
            comparisons=comparisons
        )

    def _prepare_data(
            self,
            predictions: Tensor,
            targets: Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for evaluation by unscaling if necessary."""
        # Convert to numpy and reshape if needed
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()

        # Reshape if more than 2 dimensions
        if len(pred_np.shape) > 2:
            pred_np = pred_np.reshape(-1, pred_np.shape[-1])
            target_np = target_np.reshape(-1, target_np.shape[-1])

        if self.scaler is not None:
            with contextlib.suppress(Exception):
                pred_np = self.scaler.inverse_transform(pred_np)
                target_np = self.scaler.inverse_transform(target_np)
        return pred_np, target_np

    def _create_comparisons(
            self,
            predictions: np.ndarray,
            targets: np.ndarray,
            timestamps: np.ndarray
    ) -> List[PredictionComparison]:
        """Create detailed comparisons for each timestamp."""
        return [
            PredictionComparison(
                timestamp=timestamp,
                predicted=pred,
                actual=target
            )
            for pred, target, timestamp in zip(predictions, targets, timestamps)
        ]