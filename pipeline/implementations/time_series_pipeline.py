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
        self._batch_callback = None  # Add this line

    def add_progress_callback(self, callback):
        """Add a callback function to monitor training progress.
        
        Args:
            callback: Function that takes (epoch, train_loss, val_loss) as parameters
        """
        self._progress_callback = callback

    def add_batch_callback(self, callback):
        """Add a callback function to monitor batch-level progress.
        
        Args:
            callback: Function that takes (batch_index, total_batches, loss) as parameters
        """
        self._batch_callback = callback

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

        device = self.model_wrapper.device
        self.logger.info(f"Starting model training", {"device": str(device)})

        training_report = TrainingReport(
            train_losses=[],
            val_losses=[],
            learning_rates=[],
            epochs=0,
            additional_metrics={}
        )

        # Initialize evaluator for metric calculations
        evaluator = Evaluator(
            scaler=self.config.dataset_config.time_series_scaler,
            metric_config=MetricConfig(
                resolution_minutes=self.config.dataset_config.forecast_resolution_minutes
            ),
            resolution_minutes=self.config.dataset_config.forecast_resolution_minutes
        )
        
        self.logger.debug("CUDA memory status", {
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "memory_cached_gb": torch.cuda.memory_reserved() / 1e9
        })

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
                
            epoch_predictions = []
            epoch_targets = []
            
            # Process each batch
            for batch_idx, (batch_input, batch_target) in enumerate(train_loader):
                # Move batch to correct device
                if isinstance(batch_input, tuple):
                    batch_input = tuple(b.to(device) for b in batch_input)
                else:
                    batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                
                loss = self._process_training_batch(batch_input, batch_target)
                epoch_loss += loss
                num_batches += 1

                # Call batch callback
                if self._batch_callback:
                    self._batch_callback(batch_idx, len(train_loader), float(loss))

                # Store predictions for evaluation
                with torch.no_grad():
                    if isinstance(batch_input, tuple):
                        predictions = self.model_wrapper.model(*[b.to(device) for b in batch_input])
                    else:
                        predictions = self.model_wrapper.model(batch_input.to(device))
                    epoch_predictions.append(predictions.cpu())
                    epoch_targets.append(batch_target.cpu())

                
            # Process epoch results
            avg_epoch_loss = epoch_loss / num_batches
            training_report.train_losses.append(avg_epoch_loss)
            
            # Concatenate predictions and targets
            epoch_predictions = torch.cat(epoch_predictions)
            epoch_targets = torch.cat(epoch_targets)
            
            # Validation phase
            val_loss = None
            val_predictions = None
            val_targets = None
            if val_loader:
                val_loss, val_predictions, val_targets = self._validate_model(val_loader)
                training_report.val_losses.append(val_loss)
            
            # Call progress callback if set
            if self._progress_callback:
                self._progress_callback(
                    epoch=epoch,
                    train_loss=float(avg_epoch_loss),
                    val_loss=float(val_loss) if val_loss is not None else None,
                    y_pred=val_predictions if val_predictions is not None else epoch_predictions,
                    y_true=val_targets if val_targets is not None else epoch_targets
                )
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                self.logger.info("Training progress", {
                    "epoch": epoch + 1,
                    "total_epochs": self.config.training_config.max_epochs,
                    "loss": float(avg_epoch_loss)
                })
        
        return self.model_wrapper, training_report

    def _process_training_batch(self, batch_input: Any, batch_target: torch.Tensor) -> float:
        """Process a training batch with proper gradient handling."""
        if not self.model_wrapper:
            raise ValueError("Model wrapper not initialized")

        try:
            # Handle transformer models
            if (isinstance(self.model_wrapper, PyTorchWrapper) and 
                self.config.model_config.model_type.is_transformer and 
                isinstance(batch_input, tuple)):
                
                src, tgt = batch_input
                device = self.model_wrapper.device
                dtype = self.model_wrapper.dtype
                
                # Ensure inputs have gradients enabled
                if not src.requires_grad:
                    src.requires_grad_(True)
                if not tgt.requires_grad:
                    tgt.requires_grad_(True)
                
                # Generate masks
                src_mask = self.model_wrapper.model.generate_square_subsequent_mask(src.size(1)).to(
                    device=device, dtype=dtype
                )
                tgt_mask = self.model_wrapper.model.generate_square_subsequent_mask(tgt.size(1)).to(
                    device=device, dtype=dtype
                )
                
                # Log gradient status for debugging
                self.logger.debug("Batch gradient status", {
                    "src_requires_grad": src.requires_grad,
                    "tgt_requires_grad": tgt.requires_grad,
                    "src_is_leaf": src.is_leaf,
                    "tgt_is_leaf": tgt.is_leaf
                })
                
                loss = self.model_wrapper.training_step(
                    (src, tgt), 
                    batch_target,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask,
                    use_reentrant=False  # Explicitly set checkpointing parameter
                )
                return float(loss)

            # Handle regular models
            if isinstance(batch_input, tuple):
                batch_input = batch_input[0]
                if not batch_input.requires_grad:
                    batch_input.requires_grad_(True)
                    
            loss = self.model_wrapper.training_step(batch_input, batch_target)
            return float(loss)

        except RuntimeError as e:
            error_info = {
                "error": str(e),
                "target_shape": batch_target.shape,
                "target_device": str(batch_target.device),
                "target_dtype": str(batch_target.dtype)
            }
            
            if isinstance(batch_input, tuple):
                error_info.update({
                    "src_shape": batch_input[0].shape,
                    "src_device": str(batch_input[0].device),
                    "src_requires_grad": batch_input[0].requires_grad,
                    "tgt_shape": batch_input[1].shape if len(batch_input) > 1 else None,
                    "tgt_device": str(batch_input[1].device) if len(batch_input) > 1 else None,
                    "tgt_requires_grad": batch_input[1].requires_grad if len(batch_input) > 1 else None
                })
            else:
                error_info.update({
                    "input_shape": batch_input.shape,
                    "input_device": str(batch_input.device),
                    "input_requires_grad": batch_input.requires_grad
                })
                
            self.logger.error("Training batch processing failed", error_info)
            raise
    
    def _handle_validation(self, val_loader: DataLoader, training_report: TrainingReport) -> None:
        """Handle validation phase of training."""
        val_loss, _, _ = self._validate_model(val_loader)
        if not hasattr(training_report, 'val_losses'):
            training_report.val_losses = []
        training_report.val_losses.append(float(val_loss))
        
        self.logger.debug("Validation completed", {
            "validation_loss": float(val_loss),
            "epoch": len(training_report.val_losses)
        })

    def _validate_model(self, val_loader: DataLoader) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Perform validation and return loss and predictions."""
        if self.model_wrapper is None:
            raise ValueError("Model wrapper not initialized")
        
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        # Get device for PyTorch models
        device = None
        is_transformer = False
        if isinstance(self.model_wrapper, PyTorchWrapper):
            device = next(self.model_wrapper.model.parameters()).device
            is_transformer = self.config.model_config.model_type.is_transformer

        with torch.no_grad():
            for batch_input, batch_target in val_loader:
                # Handle device placement for PyTorch models
                if device is not None:
                    if isinstance(batch_input, tuple):
                        batch_input = tuple(b.to(device) for b in batch_input)
                    else:
                        batch_input = batch_input.to(device)
                    batch_target = batch_target.to(device)

                # Get predictions
                if isinstance(self.model_wrapper, PyTorchWrapper):
                    predictions = self.model_wrapper.model(batch_input) if not is_transformer else \
                                self.model_wrapper.model(*batch_input)
                else:
                    # For sklearn models
                    if isinstance(batch_input, tuple):
                        batch_input = batch_input[0]
                    predictions = torch.tensor(self.model_wrapper.model.predict(batch_input.numpy()))

                # Calculate and accumulate loss
                loss = self.model_wrapper.validation_step(batch_input, batch_target)
                total_loss += float(loss)  # Ensure we get a float
                num_batches += 1

                # Store predictions and targets (move to CPU if needed)
                all_predictions.append(predictions.cpu() if device is not None else predictions)
                all_targets.append(batch_target.cpu() if device is not None else batch_target)

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        predictions = torch.cat(all_predictions) if all_predictions else None
        targets = torch.cat(all_targets) if all_targets else None

        return avg_loss, predictions, targets

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