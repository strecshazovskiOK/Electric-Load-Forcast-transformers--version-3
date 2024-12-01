# experiments/utils/analysis.py
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from ..experiment import Experiment

class ExperimentAnalyzer:
    """Utility class for analyzing experiment results."""

    @staticmethod
    def compare_experiments(experiments: List[Experiment]) -> pd.DataFrame:
        """Compare multiple experiments and their metrics."""
        comparisons = []
        for exp in experiments:
            metrics = exp.evaluation.total_metrics
            comparison = {
                'experiment_name': str(exp.model_wrapper.model_type),
                'timestamp': datetime.now(),
                'training_time': exp.training_time,
                'test_time': exp.test_time,
                **metrics
            }
            comparisons.append(comparison)
        return pd.DataFrame(comparisons)

    @staticmethod
    def plot_training_curves(experiment: Experiment) -> None:
        """Plot training and validation loss curves."""
        if experiment.training_report is None:
            print("No training report available")
            return

        train_losses = experiment.training_report.train_losses
        val_losses = experiment.training_report.val_losses
        epochs = list(range(len(train_losses)))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Training Loss')
        if val_losses:  # Only plot validation losses if they exist
            plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Progress - {experiment.model_wrapper.model_type}')
        plt.legend()
        plt.grid(True)
        plt.show()