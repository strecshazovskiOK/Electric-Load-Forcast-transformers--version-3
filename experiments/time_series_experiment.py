# experiments/time_series_experiment.py
import datetime
import json
import os
from typing import Optional, Dict, Any

from models.base.base_wrapper import BaseWrapper
from training.reports.training_report import TrainingReport
from evaluation.results import EvaluationResult  # Changed from Evaluation to EvaluationResult

from .experiment import Experiment

class TimeSeriesExperiment(Experiment):
    """
    Specialized experiment implementation for time series forecasting.
    Extends base Experiment class with time series specific functionality.
    """

    def __init__(
            self,
            model_wrapper: BaseWrapper,
            evaluation: EvaluationResult,  # Changed from Evaluation to EvaluationResult
            training_config,
            training_report: Optional[TrainingReport],  # Made training_report optional
            training_time: float,
            test_time: float,
            forecasting_horizon: Optional[int] = None,
            time_series_window: Optional[int] = None
    ):
        super().__init__(
            model_wrapper,
            evaluation,
            training_config,
            training_report,
            training_time,
            test_time
        )
        self.forecasting_horizon = forecasting_horizon
        self.time_series_window = time_series_window

    def save_to_json_file(self) -> None:
        """
        Extends base save functionality with time series specific metrics.
        """
        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        experiment_name = f"{str(self.model_wrapper.model_type)}_{date}"
        print(experiment_name)

        result = {
            'experimentName': experiment_name,
            'modelType': str(self.model_wrapper.model_type),
            'modelWrapper': str(self.model_wrapper),
            'trainingConfig': self.training_config.__dict__,
            'trainingReport': self.training_report.serialize() if self.training_report else None,
            'evaluation': self.evaluation.serialize(),
            'training_time': self.training_time,
            'test_time': self.test_time,
            # Add time series specific information
            'forecasting_horizon': self.forecasting_horizon,
            'time_series_window': self.time_series_window
        }

        file_path = self._get_experiment_filepath(experiment_name)
        self._save_json(file_path, result)

    def _get_experiment_filepath(self, experiment_name: str) -> str:
        """Get the complete file path for the experiment."""
        return self._create_experiment_path(
            self.EXPERIMENTS_DIRECTORY,
            self.FINAL_EXPERIMENTS_DIRECTORY,
            f"{experiment_name}{self.JSON_FILE_ENDING}"
        )

    def _save_json(self, filepath: str, data: Dict[str, Any]) -> None:
        """Save data as JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as fp:
            json.dump(data, fp, indent=4)

    def _create_experiment_path(self, root_dir: str, subdir: str, filename: str) -> str:
        """Create complete experiment file path."""
        path = os.path.join(root_dir, subdir)
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, filename)

    @staticmethod
    def load_from_file(filepath: str) -> 'TimeSeriesExperiment':
        """
        Load a time series experiment from a JSON file.

        Args:
            filepath: Path to the experiment JSON file

        Returns:
            TimeSeriesExperiment instance
        """
        with open(filepath, 'r') as fp:
            data = json.load(fp)

        training_report_data = data['trainingReport']
        training_report = TrainingReport.deserialize(training_report_data) if training_report_data else None

        return TimeSeriesExperiment(
            model_wrapper=data['modelWrapper'],
            evaluation=EvaluationResult.from_dict(data['evaluation']),  # Changed to use from_dict
            training_config=data['trainingConfig'],
            training_report=training_report,
            training_time=data['training_time'],
            test_time=data['test_time'],
            forecasting_horizon=data.get('forecasting_horizon'),
            time_series_window=data.get('time_series_window')
        )

    def __str__(self):
        base_str = super().__str__()
        ts_specific = (
            f'Forecasting Horizon: {self.forecasting_horizon}\n'
            f'Time Series Window: {self.time_series_window}'
        )
        return f"{base_str}\n{ts_specific}"