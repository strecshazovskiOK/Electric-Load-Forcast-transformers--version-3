# pipeline/base/base_pipeline.py
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Optional, Tuple
from pathlib import Path

from torch.utils.data import Dataset
from torch.optim.optimizer import Optimizer
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW




from data_loading.loaders.time_series_loader import TimeInterval, TimeSeriesLoader

from ..config.pipeline_config import PipelineConfig
from experiments.experiment import Experiment


class BasePipeline(ABC):
    """Base class for all pipeline implementations."""

    def __init__(self, config: PipelineConfig):
        """
        Args:
            config: Configuration for dataset, model, and training
        """
        self.config = config
        self.data_loader: Optional[TimeSeriesLoader] = None
        self.experiment: Optional[Experiment] = None
        self.training_time: float = 0.0
        self.test_time: float = 0.0
        
        self.config_dict = asdict(self.config)


    @abstractmethod
    def prepare_data_loader(self) -> TimeSeriesLoader:
        """Initialize and prepare the data loader."""
        pass

    @abstractmethod
    def prepare_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare train, validation and test datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        pass

    @abstractmethod
    def setup_model(self) -> None:
        """Setup model and its wrapper."""
        pass

    @abstractmethod
    def train_model(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        pass

    @abstractmethod
    def evaluate_model(self, test_dataset: Dataset) -> None:
        """
        Evaluate model performance.

        Args:
            test_dataset: Test dataset
        """
        pass

    def run(self) -> Optional[Experiment]:
        """
        Execute the complete pipeline.

        Returns:
            Experiment object containing results and metrics
        """
        try:
            # Initialize data loader
            self.data_loader = self.prepare_data_loader()

            # Prepare datasets
            train_dataset, val_dataset, test_dataset = self.prepare_datasets()

            # Setup and train model
            self.setup_model()
            self.train_model(train_dataset, val_dataset)

            # Evaluate model
            self.evaluate_model(test_dataset)

            # Save experiment if path provided
            if self.experiment and self.config.experiment_save_path:
                self.experiment.save_to_json_file()

            return self.experiment

        except Exception as e:
            print(f"Pipeline execution failed: {str(e)}")
            raise

    def _create_time_intervals(self) -> Tuple[TimeInterval, TimeInterval, TimeInterval]:
        """Create time intervals for data splitting."""
        return (
            TimeInterval(
                min_date=self.config.train_dates[0],
                max_date=self.config.train_dates[1]
            ),
            TimeInterval(
                min_date=self.config.val_dates[0],
                max_date=self.config.val_dates[1]
            ),
            TimeInterval(
                min_date=self.config.test_dates[0],
                max_date=self.config.test_dates[1]
            )
        )

    def _save_model(self, save_path: Path) -> None:
        """
        Save model artifacts.

        Args:
            save_path: Path to save model files
        """
        pass
        
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Helper method to safely get configuration values."""
        return self.config_dict.get(key, default)
