from __future__ import annotations

import time
from typing import Tuple, Optional, Dict, Any, cast, Union
from pathlib import Path
import torch
import dataclasses
from argparse import Namespace
import numpy as np
from numpy.typing import NDArray

from data_loading.datasets.standard_dataset import StandardDataset
from data_loading.datasets.transformer_dataset import TransformerDataset
from data_loading.loaders.time_series_loader import TimeSeriesLoader, TimeInterval
from evaluation.evaluator import Evaluator
from experiments.experiment import Experiment
from models.base.base_wrapper import BaseWrapper
from models.registry.factory import ModelFactory
from models.registry.model_types import ModelType
from models.wrappers.pytorch_wrapper import PyTorchWrapper
from models.wrappers.sklearn_wrapper import SklearnWrapper
from pipeline.base.base_pipeline import BasePipeline

class TimeSeriesPipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader: Optional[TimeSeriesLoader] = None
        self.model_wrapper: Optional[Union[SklearnWrapper, PyTorchWrapper]] = None

    def prepare_data_loader(self) -> TimeSeriesLoader:
        """Initialize and configure the time series data loader."""
        self.data_loader = TimeSeriesLoader(
            time_variable=self.config.dataset_config.time_variable,
            target_variable=self.config.dataset_config.target_variable
        )
        return self.data_loader

    def prepare_datasets(self) -> Tuple[StandardDataset | TransformerDataset, ...]:
        """Prepare train, validation and test datasets."""
        if not self.data_loader:
            raise ValueError("Data loader not initialized")

        # Load data
        df = self.data_loader.load(self.config.data_path)

        # Get time intervals
        train_interval, val_interval, test_interval = self._create_time_intervals()

        # Split data
        train_df, val_df, test_df = self.data_loader.split(
            df,  # Add df as first argument
            train_interval,
            val_interval,
            test_interval
        )

        # Create appropriate dataset types based on model
        dataset_cls = (
            TransformerDataset
            if self.config.model_config.model_type.is_transformer
            else StandardDataset
        )

        # Create datasets
        train_dataset = dataset_cls(train_df, self.config.dataset_config)
        val_dataset = dataset_cls(
            val_df,
            dataclasses.replace(
                self.config.dataset_config,
                is_training_set=False
            )
        )
        test_dataset = dataset_cls(
            test_df,
            dataclasses.replace(
                self.config.dataset_config,
                is_training_set=False
            )
        )

        return train_dataset, val_dataset, test_dataset

    def setup_model(self) -> None:
        """Setup model and associated wrapper."""
        # Convert config to dict for factory
        model_config_dict = dataclasses.asdict(self.config.model_config)

        # Create model using factory
        model = ModelFactory.create_base_model(
            self.config.model_config.model_type,
            model_config_dict
        )

        # Convert training config to dict
        training_config_dict = dataclasses.asdict(self.config.training_config)

        # Create appropriate wrapper
        if self.config.model_config.model_type == ModelType.LINEAR_REGRESSION:
            wrapper: Union[SklearnWrapper, PyTorchWrapper] = SklearnWrapper(
                cast(Any, model),
                self.config.model_config.model_type,
                training_config_dict
            )
        else:
            wrapper = PyTorchWrapper(
                model,
                self.config.model_config.model_type,
                training_config_dict
            )
        
        # Assign wrapper with proper type annotation
        self.model_wrapper = wrapper

    def train_model(self, train_dataset: StandardDataset | TransformerDataset,
                    val_dataset: StandardDataset | TransformerDataset) -> None:
        """Train the model and record metrics."""
        if not self.model_wrapper:
            raise ValueError("Model wrapper not initialized")

        training_start = time.time()

        self.training_report = self.model_wrapper.train(
            train_dataset,
            val_dataset
        )

        self.training_time = time.time() - training_start

    def evaluate_model(self, test_dataset: StandardDataset | TransformerDataset) -> None:
        """Evaluate model and create experiment."""
        if not self.model_wrapper:
            raise ValueError("Model wrapper not initialized")

        # Make predictions
        test_start = time.time()
        predictions, targets = self.model_wrapper.predict(test_dataset)
        self.test_time = time.time() - test_start

        # Get timestamps with null check
        timestamps = test_dataset.time_labels
        if timestamps is None:
            timestamps = np.arange(len(predictions))  # Fallback to indices if no timestamps

        # Create evaluator with the dataset's scaler
        evaluator = Evaluator(
            scaler=self.config.dataset_config.time_series_scaler
        )

        # Evaluate predictions
        evaluation = evaluator.evaluate(
            predictions=predictions,
            targets=targets,
            timestamps=cast(NDArray, timestamps),  # Cast to satisfy type checker
            num_variables=1  # Assuming single variable prediction
        )

        # Create experiment with type-safe arguments
        self.experiment = Experiment(
            model_wrapper=cast(BaseWrapper, self.model_wrapper),
            evaluation=evaluation,
            training_config=Namespace(**dataclasses.asdict(self.config.training_config)),
            training_report=self.training_report,
            training_time=self.training_time,
            test_time=self.test_time
        )

        # Save model if path provided
        if self.config.model_save_path:
            self._save_model(str(self.config.model_save_path))

    def _save_model(self, save_path: str) -> None:
        """Save model artifacts."""
        if not self.model_wrapper:
            raise ValueError("Model wrapper not initialized")

        # Save PyTorch model
        if isinstance(self.model_wrapper, PyTorchWrapper):
            if hasattr(self.model_wrapper.model, 'state_dict'):
                torch.save(
                    self.model_wrapper.model.state_dict(),
                    f"{save_path}/model.pt"
                )

            # Save scaler if exists
            if self.config.dataset_config.time_series_scaler:
                import pickle
                with open(f"{save_path}/scaler.pkl", "wb") as f:
                    pickle.dump(
                        self.config.dataset_config.time_series_scaler,
                        f
                    )