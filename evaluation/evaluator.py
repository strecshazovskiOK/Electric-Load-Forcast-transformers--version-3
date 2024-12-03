from typing import Optional, Tuple, List
import numpy as np
import torch
from torch import Tensor
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from evaluation.metrics import Metrics, MetricConfig
from evaluation.results import EvaluationResult, PredictionComparison
from utils.logging.config import LogLevel, LoggerConfig
from utils.logging.logger import Logger  # Assuming metrics are in evaluation package

# evaluation/evaluator.py

class Evaluator:
    """Evaluates model predictions against actual values."""

    def __init__(
            self,
            scaler: Optional[StandardScaler] = None,
            metric_config: Optional[MetricConfig] = None,
            resolution_minutes: Optional[int] = None
    ):
        self.logger = Logger.get_logger(
            __name__,
            LoggerConfig(
                level=LogLevel.INFO,
                component_name="evaluator",
                file_path=Path("logs/evaluation.log"),
                json_output=True
            )
        )
        
        self.logger.info(
            "Initializing evaluator",
            extra={
                "resolution_minutes": resolution_minutes,
                "has_scaler": scaler is not None,
                "metric_config": metric_config.__dict__ if metric_config else None
            }
        )
        
        self.scaler = scaler
        self.metric_config = metric_config or MetricConfig()
        self.resolution_minutes = resolution_minutes

    def evaluate(
            self,
            predictions: Tensor,
            targets: Tensor,
            timestamps: np.ndarray,
            num_variables: int = 1
    ) -> EvaluationResult:
        """Evaluate model predictions."""
        self.logger.info(
            "Starting evaluation",
            extra={
                "predictions_shape": predictions.shape,
                "targets_shape": targets.shape,
                "num_variables": num_variables
            }
        )

        try:
            # Unscale predictions if scaler is provided
            pred_np, target_np = self._prepare_data(predictions, targets)
            self.logger.debug(
                "Data prepared for evaluation",
                extra={
                    "pred_range": [float(pred_np.min()), float(pred_np.max())],
                    "target_range": [float(target_np.min()), float(target_np.max())]
                }
            )

            # Calculate overall metrics
            total_metrics = Metrics.calculate_all_metrics(
                torch.tensor(pred_np),
                torch.tensor(target_np),
                self.metric_config
            )
            self.logger.info("Overall metrics calculated", extra={"metrics": total_metrics})

            # Calculate per-variable metrics if multiple variables
            variable_metrics = {}
            if num_variables > 1:
                for i in range(num_variables):
                    variable_metrics[f'var_{i}'] = Metrics.calculate_all_metrics(
                        torch.tensor(pred_np[:, i]),
                        torch.tensor(target_np[:, i]),
                        self.metric_config
                    )
                self.logger.info(
                    "Variable-specific metrics calculated", 
                    extra={"variable_metrics": variable_metrics}
                )

            # Create detailed comparisons
            comparisons = self._create_comparisons(pred_np, target_np, timestamps)
            self.logger.debug(f"Created {len(comparisons)} detailed comparisons")

            result = EvaluationResult(
                total_metrics=total_metrics,
                variable_metrics=variable_metrics,
                comparisons=comparisons
            )
            
            self.logger.info("Evaluation completed successfully")
            return result

        except Exception as e:
            self.logger.error(
                "Evaluation failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise

    def _prepare_data(
            self,
            predictions: Tensor,
            targets: Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for evaluation."""
        self.logger.debug("Preparing data for evaluation")
        
        # Convert to numpy and reshape if needed
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()

        if self.scaler is not None:
            try:
                pred_np = self.scaler.inverse_transform(pred_np)
                target_np = self.scaler.inverse_transform(target_np)
                self.logger.debug("Data unscaled successfully")
            except Exception as e:
                self.logger.warning(
                    "Failed to unscale data",
                    extra={"error": str(e)}
                )

        return pred_np, target_np

    def _create_comparisons(
            self,
            predictions: np.ndarray,
            targets: np.ndarray,
            timestamps: np.ndarray
    ) -> List[PredictionComparison]:
        """Create detailed comparisons for each timestamp."""
        self.logger.debug("Creating detailed comparisons")
        
        try:
            comparisons = [
                PredictionComparison(
                    timestamp=timestamp,
                    predicted=pred,
                    actual=target
                )
                for pred, target, timestamp in zip(predictions, targets, timestamps)
            ]
            
            self.logger.debug(
                f"Created {len(comparisons)} comparison records",
                extra={
                    "first_timestamp": str(comparisons[0].timestamp),
                    "last_timestamp": str(comparisons[-1].timestamp)
                }
            )
            
            return comparisons
            
        except Exception as e:
            self.logger.error(
                "Failed to create comparisons",
                extra={"error": str(e)}
            )
            raise