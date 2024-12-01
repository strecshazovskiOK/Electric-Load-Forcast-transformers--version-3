# experiments/utils/loader.py
import os
import json
from typing import List, Optional
from datetime import datetime
from ..experiment import Experiment

class ExperimentLoader:
    """Utility class for loading and managing experiment results."""

    def __init__(self, experiments_dir: str = 'experiments'):
        self.experiments_dir = experiments_dir

    def load_experiment(self, experiment_id: str) -> Experiment:
        """Load a specific experiment by ID."""
        filepath = os.path.join(self.experiments_dir, f"{experiment_id}.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Experiment {experiment_id} not found")

        with open(filepath, 'r') as f:
            data = json.load(f)
        return Experiment.from_json(data)

    def list_experiments(
            self,
            model_type: Optional[str] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None
    ) -> List[str]:
        """List experiment IDs matching criteria."""
        experiments = []

        for filename in os.listdir(self.experiments_dir):
            if not filename.endswith('.json'):
                continue

            with open(os.path.join(self.experiments_dir, filename), 'r') as f:
                data = json.load(f)

            # Apply filters
            if model_type and data['modelType'] != model_type:
                continue

            exp_date = datetime.fromisoformat(data['experimentName'].split('_', 1)[1])
            if start_date and exp_date < start_date:
                continue
            if end_date and exp_date > end_date:
                continue

            experiments.append(filename[:-5])  # Remove .json extension

        return sorted(experiments)

    def get_best_experiment(
            self,
            metric: str = 'mape',
            model_type: Optional[str] = None
    ) -> Experiment:
        """Get the best performing experiment based on specified metric."""
        experiments = self.list_experiments(model_type=model_type)

        best_score = float('inf')
        best_experiment = None

        for exp_id in experiments:
            experiment = self.load_experiment(exp_id)
            score = experiment.evaluation.total_metrics.get(metric)

            if score is not None and score < best_score:
                best_score = score
                best_experiment = experiment

        if best_experiment is None:
            raise ValueError(f"No experiments found with metric {metric}")

        return best_experiment