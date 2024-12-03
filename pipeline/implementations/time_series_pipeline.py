# pipeline/implementations/time_series_pipeline.py
from __future__ import annotations
import time
from typing import Tuple, Optional, Dict, Any, cast, Union
from pathlib import Path
import torch
import dataclasses
from argparse import Namespace
import numpy as np
from numpy.typing import NDArray
import pandas as pd  # Add this import statement
from datetime import date  # Add this import statement

from data_loading.datasets.standard_dataset import StandardDataset
from data_loading.datasets.transformer_dataset import TransformerDataset
from data_loading.loaders.time_series_loader import TimeSeriesLoader, TimeInterval
from evaluation.evaluator import Evaluator
from evaluation.metrics import MetricConfig
from experiments.experiment import Experiment
from models.base.base_wrapper import BaseWrapper
from models.registry.factory import ModelFactory
from models.registry.model_types import ModelType
from models.wrappers.pytorch_wrapper import PyTorchWrapper
from models.wrappers.sklearn_wrapper import SklearnWrapper
from pipeline.base.base_pipeline import BasePipeline
from training.reports.training_report import TrainingReport
from torch.utils.data import DataLoader
from utils.logging.logger import Logger
from utils.logging.config import LoggerConfig, LogLevel

class TimeSeriesPipeline(BasePipeline):
    """Pipeline implementation for time series forecasting with resolution awareness."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader: Optional[TimeSeriesLoader] = None
        self.model_wrapper: Optional[Union[SklearnWrapper, PyTorchWrapper]] = None
        
        # Initialize logger
        logger_config = LoggerConfig(
            level=LogLevel.INFO,
            component_name="TimeSeriesPipeline",
            include_timestamp=True
        )
        self.logger = Logger.get_logger(__name__, logger_config)
        
        # Validate resolutions during initialization
        self._validate_resolutions()
        self._progress_callback = None

    def add_progress_callback(self, callback):
        """Add a callback function to monitor training progress.
        
        Args:
            callback: Function that takes (epoch, train_loss, val_loss) as parameters
        """
        self._progress_callback = callback

    def _validate_resolutions(self) -> None:
        """Validate that resolutions are consistent across configurations."""
        input_res = self.config.dataset_config.input_resolution_minutes
        forecast_res = self.config.dataset_config.forecast_resolution_minutes
        
        # Check model type matches resolution
        if self.config.model_config.model_type.is_resolution_specific:
            expected_type = ModelType.get_for_resolution(forecast_res)
            if self.config.model_config.model_type != expected_type:
                raise ValueError(
                    f"Model type {self.config.model_config.model_type} does not match "
                    f"forecast resolution {forecast_res} minutes. Expected {expected_type}"
                )

    def prepare_data_loader(self) -> TimeSeriesLoader:
        """Initialize and configure the time series data loader with resolution awareness."""
        self.data_loader = TimeSeriesLoader(
            time_variable=self.config.dataset_config.time_variable,
            target_variable=self.config.dataset_config.target_variable
        )
        return self.data_loader

    def prepare_datasets(self) -> Tuple[StandardDataset | TransformerDataset, ...]:
        """Prepare train, validation and test datasets with resolution handling."""
        if not self.data_loader:
            raise ValueError("Data loader not initialized")

        # Load and potentially resample data
        df = self.data_loader.load(self.config.data_path)
        
        # Resample if input and forecast resolutions differ
        if self.config.dataset_config.needs_resampling():
            df = self._resample_data(
                df,
                target_resolution_minutes=self.config.dataset_config.forecast_resolution_minutes
            )

        # Get time intervals adjusted for resolution
        train_interval, val_interval, test_interval = self._create_resolution_aware_intervals()

        # Split data
        train_df, val_df, test_df = self.data_loader.split(
            df,
            train_interval,
            val_interval,
            test_interval
        )

        # Select appropriate dataset class
        dataset_cls = self._get_dataset_class()

        # Create datasets
        train_dataset = dataset_cls(train_df, self.config.dataset_config)
        val_dataset = dataset_cls(
            val_df,
            dataclasses.replace(self.config.dataset_config, is_training_set=False)
        )
        test_dataset = dataset_cls(
            test_df,
            dataclasses.replace(self.config.dataset_config, is_training_set=False)
        )

        return train_dataset, val_dataset, test_dataset

    def _resample_data(self, df: pd.DataFrame, target_resolution_minutes: int) -> pd.DataFrame:
        """Resample data to match the target resolution."""
        if self.data_loader is None:
            raise ValueError("Data loader not initialized")
        rule = f'{target_resolution_minutes}T'
        resampled = df.resample(rule, on=self.data_loader.time_variable).mean()
        return resampled.reset_index()

    def _create_resolution_aware_intervals(self) -> Tuple[TimeInterval, TimeInterval, TimeInterval]:
        if self.config.dataset_config.forecast_resolution_minutes >= 43200:  # Monthly
            return self._create_monthly_aligned_intervals()
        
        # For daily predictions, ensure intervals align with day boundaries
        elif self.config.dataset_config.forecast_resolution_minutes >= 1440:  # Daily
            return self._create_daily_aligned_intervals()
            
        # For hourly and sub-hourly, use standard intervals
        return self._create_time_intervals()

    def _create_monthly_aligned_intervals(self) -> Tuple[TimeInterval, TimeInterval, TimeInterval]:
        """Create monthly aligned intervals."""
        train_start, train_end = self.config.train_dates
        val_start, val_end = self.config.val_dates
        test_start, test_end = self.config.test_dates
        
        return (
            TimeInterval(min_date=train_start, max_date=train_end),
            TimeInterval(min_date=val_start, max_date=val_end),
            TimeInterval(min_date=test_start, max_date=test_end)
        )

    def _create_daily_aligned_intervals(self) -> Tuple[TimeInterval, TimeInterval, TimeInterval]:
        """Create daily aligned intervals."""
        train_start, train_end = self.config.train_dates
        val_start, val_end = self.config.val_dates
        test_start, test_end = self.config.test_dates
        
        return (
            TimeInterval(min_date=train_start, max_date=train_end),
            TimeInterval(min_date=val_start, max_date=val_end),
            TimeInterval(min_date=test_start, max_date=test_end)
        )

    def _create_time_intervals(self) -> Tuple[TimeInterval, TimeInterval, TimeInterval]:
        """Create standard time intervals."""
        return self._create_monthly_aligned_intervals()

    def _get_dataset_class(self):
        """Get appropriate dataset class based on model type and resolution."""
        if self.config.model_config.model_type.is_transformer:
            if self.config.model_config.model_type.is_resolution_specific:
                # Use TransformerDataset with resolution-specific optimizations
                return TransformerDataset
            return TransformerDataset
        return StandardDataset

    def setup_model(self) -> None:
        """Setup model and associated wrapper with resolution awareness."""
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
        
        self.model_wrapper = wrapper

        self.logger.debug("Model setup complete", {
            "device_config": training_config_dict.get('device', 'Not specified'),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        })

    def evaluate_model(self, test_dataset: StandardDataset | TransformerDataset) -> None:
        """Evaluate model with resolution-aware metrics."""
        if not self.model_wrapper:
            raise ValueError("Model wrapper not initialized")

        # Make predictions
        test_start = time.time()
        predictions, targets = self.model_wrapper.predict(test_dataset)
        self.test_time = time.time() - test_start

        # Get timestamps
        timestamps = test_dataset.time_labels
        if timestamps is None:
            timestamps = np.arange(len(predictions))

        # Create evaluator with resolution-aware configuration
        evaluator = Evaluator(
            scaler=self.config.dataset_config.time_series_scaler,
            metric_config=MetricConfig(),  # Add this if needed
            resolution_minutes=self.config.dataset_config.forecast_resolution_minutes
        )

        # Evaluate predictions
        evaluation = evaluator.evaluate(
            predictions=predictions,
            targets=targets,
            timestamps=cast(NDArray, timestamps),
            num_variables=1
        )

        # Create experiment
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
            self._save_model(self.config.model_save_path)

    @property
    def training_report(self) -> TrainingReport:  # Change return type
        """Get training report."""
        if not hasattr(self, '_training_report'):
            self._training_report = TrainingReport(train_losses=[])  # Initialize with empty train_losses list
        return self._training_report

    def train_model(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Tuple[PyTorchWrapper, TrainingReport]:
        """Train model implementation."""
        if not self.model_wrapper:
            raise ValueError("Model wrapper not initialized")
        if not isinstance(self.model_wrapper, PyTorchWrapper):
            raise TypeError("Model wrapper must be PyTorchWrapper for time series training")

        # Get device from model wrapper
        device = self.model_wrapper.device
        self.logger.info(f"Starting model training", {"device": str(device)})

        # Initialize training report with metrics
        training_report = TrainingReport(
            train_losses=[],
            val_losses=[],
            learning_rates=[],
            epochs=0,
            additional_metrics={}
        )
        
        self.logger.debug("CUDA memory status", {
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "memory_cached_gb": torch.cuda.memory_reserved() / 1e9
        })
        
        # Monitor device placement during training
        for epoch in range(self.config.training_config.max_epochs):
            if epoch == 0 or epoch % 10 == 0:  # Print every 10 epochs
                self.logger.debug(f"Epoch status", {
                    "epoch": epoch,
                    "model_device": str(next(self.model_wrapper.model.parameters()).device),
                    "cuda_memory_gb": torch.cuda.memory_allocated() / 1e9
                })

            epoch_loss = 0.0
            num_batches = 0
            
            # Set model to training mode
            if hasattr(self.model_wrapper.model, 'train'):
                self.model_wrapper.model.train()
                
            # Process each batch
            for batch_input, batch_target in train_loader:
                # Move batch to correct device
                if isinstance(batch_input, tuple):
                    batch_input = tuple(b.to(device) for b in batch_input)
                else:
                    batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                
                loss = self._process_training_batch(batch_input, batch_target)
                epoch_loss += loss
                num_batches += 1

                
            # Process epoch results
            avg_epoch_loss = epoch_loss / num_batches
            training_report.train_losses.append(avg_epoch_loss)
            
            # Validation phase
            val_loss = None
            if val_loader:
                val_loss = self._handle_validation(val_loader, training_report)
                if self.config.training_config.use_early_stopping and self._should_stop_early(training_report):
                    break
            
            # Call progress callback if set
            if self._progress_callback:
                self._progress_callback(epoch, avg_epoch_loss, val_loss)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                self.logger.info("Training progress", {
                    "epoch": epoch + 1,
                    "total_epochs": self.config.training_config.max_epochs,
                    "loss": float(avg_epoch_loss)
                })
        
        return self.model_wrapper, training_report

    def _process_training_batch(self, batch_input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                            batch_target: torch.Tensor) -> float:
        """Process a single training batch with device tracking."""
        if self.model_wrapper is None:
            raise ValueError("Model wrapper not initialized")
        
        # Debug logging for device placement
        debug_info = {
            "target_device": str(batch_target.device),
            "model_device": str(next(self.model_wrapper.model.parameters()).device) if isinstance(self.model_wrapper, PyTorchWrapper) else "N/A"
        }
        
        if isinstance(batch_input, tuple):
            debug_info.update({
                "source_input_device": str(batch_input[0].device),
                "target_input_device": str(batch_input[1].device)
            })
        else:
            debug_info["input_device"] = str(batch_input.device)
            
        self.logger.debug("Batch processing details", debug_info)
        
        try:
            # Handle PyTorch models
            if isinstance(self.model_wrapper, PyTorchWrapper):
                if isinstance(batch_input, tuple):
                    src, tgt = batch_input
                    if self.config.model_config.model_type.is_transformer:
                        # Generate masks on correct device
                        device = next(self.model_wrapper.model.parameters()).device
                        src_mask = self.model_wrapper.model.generate_square_subsequent_mask(src.size(1)).to(device)
                        tgt_mask = self.model_wrapper.model.generate_square_subsequent_mask(tgt.size(1)).to(device)
                        return self.model_wrapper.training_step((src, tgt), batch_target, src_mask=src_mask, tgt_mask=tgt_mask)
                    batch_input = batch_input[0]
                return self.model_wrapper.training_step(batch_input, batch_target)
            
            # Handle sklearn models
            else:
                if isinstance(batch_input, tuple):
                    batch_input = batch_input[0]
                return self.model_wrapper.training_step(batch_input, batch_target)
                
        except RuntimeError as e:
            error_info = {
                "error": str(e),
                "target_shape": batch_target.shape,
                "target_device": str(batch_target.device)
            }
            
            if isinstance(batch_input, tuple):
                error_info.update({
                    "source_shape": batch_input[0].shape,
                    "source_device": str(batch_input[0].device),
                    "target_input_shape": batch_input[1].shape,
                    "target_input_device": str(batch_input[1].device)
                })
            else:
                error_info.update({
                    "input_shape": batch_input.shape,
                    "input_device": str(batch_input.device)
                })
                
            if isinstance(self.model_wrapper, PyTorchWrapper):
                error_info["model_device"] = str(next(self.model_wrapper.model.parameters()).device)
                
            self.logger.error("Training batch processing failed", error_info)
            raise
    
    def _handle_validation(self, val_loader: DataLoader, training_report: TrainingReport) -> None:
        """Handle validation phase of training."""
        val_loss = self._validate_model(val_loader)
        if not hasattr(training_report, 'val_losses'):
            training_report.val_losses = []
        training_report.val_losses.append(val_loss)
        
        self.logger.debug("Validation completed", {
            "validation_loss": float(val_loss),
            "epoch": len(training_report.val_losses)
        })

    def _validate_model(self, val_loader: DataLoader) -> float:
        """Perform validation."""
        if self.model_wrapper is None:
            raise ValueError("Model wrapper not initialized")
        
        total_loss = 0.0
        num_batches = 0
        is_transformer = self.config.model_config.model_type.is_transformer

        # Add validation device debugging
        debug_info = {}
        if isinstance(self.model_wrapper, PyTorchWrapper):
            debug_info["model_device"] = str(next(self.model_wrapper.model.parameters()).device)
        
        self.logger.debug("Starting validation", debug_info)
        
        for batch_input, batch_target in val_loader:
            if isinstance(batch_input, tuple):
                debug_info.update({
                    "source_validation_device": str(batch_input[0].device),
                    "target_validation_device": str(batch_input[1].device)
                })
            else:
                debug_info["input_validation_device"] = str(batch_input.device)
            debug_info["target_validation_device"] = str(batch_target.device)
            break  # Only print first batch

        self.logger.debug("Validation batch devices", debug_info)

        with torch.no_grad():
            for batch_input, batch_target in val_loader:
                # Handle PyTorch models
                if isinstance(self.model_wrapper, PyTorchWrapper):
                    if is_transformer:
                        if not isinstance(batch_input, tuple):
                            raise ValueError("Transformer models expect a tuple of (src, tgt) sequences")
                        src, tgt = batch_input
                        device = next(self.model_wrapper.model.parameters()).device
                        src = src.to(device)
                        tgt = tgt.to(device)
                        batch_target = batch_target.to(device)
                        loss = self.model_wrapper.validation_step((src, tgt), batch_target)
                    else:
                        if isinstance(batch_input, tuple):
                            batch_input = batch_input[0]
                        device = next(self.model_wrapper.model.parameters()).device
                        batch_input = batch_input.to(device)
                        batch_target = batch_target.to(device)
                        loss = self.model_wrapper.validation_step(batch_input, batch_target)
                # Handle sklearn models
                else:
                    if isinstance(batch_input, tuple):
                        batch_input = batch_input[0]
                    loss = self.model_wrapper.validation_step(batch_input, batch_target)
                    
                total_loss += loss
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')


    def _prepare_validation_data(self, val_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare validation data for sklearn models."""
        X_list, y_list = [], []
        for batch_input, batch_target in val_loader:
            X_list.append(batch_input.numpy())
            y_list.append(batch_target.numpy())
        return np.vstack(X_list), np.vstack(y_list)

    def _should_stop_early(self, training_report: TrainingReport) -> bool:
        """Check if training should stop based on validation performance."""
        if len(training_report.val_losses) < self.config.training_config.early_stopping_patience:
            return False
            
        patience = self.config.training_config.early_stopping_patience
        recent_losses = training_report.val_losses[-patience:]
        min_loss_idx = recent_losses.index(min(recent_losses))
        
        return min_loss_idx == 0  # Stop if best loss was patience steps ago

    def training_step(
        self, 
        batch_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]], 
        batch_target: torch.Tensor
    ) -> float:
        """Modified training step to handle transformer input format"""
        if self.model_wrapper is None:
            raise ValueError("Model wrapper not initialized")
        
        if isinstance(batch_input, tuple):
            if self.config.model_config.model_type.is_transformer:
                src = batch_input[0]
                tgt = batch_target[:, :-1]
                tgt_y = batch_target[:, 1:]
                return cast(float, self.model_wrapper.training_step((src, tgt), tgt_y))
            batch_input = batch_input[0]
            
        return cast(float, self.model_wrapper.training_step(batch_input, batch_target))