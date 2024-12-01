# experiments/base/base_experiment.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    model_type: str
    training_config: Dict[str, Any]
    evaluation_config: Dict[str, Any]
    experiment_name: Optional[str] = None
    save_dir: str = 'experiments'

@dataclass
class ExperimentResult:
    """Results from an experiment run."""
    model_type: str
    model_config: Dict[str, Any]
    training_report: Dict[str, Any]
    evaluation_metrics: Dict[str, Any]
    training_time: float
    inference_time: float
    timestamp: datetime

class BaseExperiment(ABC):
    """Base class for all experiments."""

    @abstractmethod
    def run(self) -> ExperimentResult:
        """Run the experiment."""
        pass

    @abstractmethod
    def save(self, result: ExperimentResult) -> None:
        """Save experiment results."""
        pass

    @abstractmethod
    def load(self, experiment_id: str) -> ExperimentResult:
        """Load experiment results."""
        pass