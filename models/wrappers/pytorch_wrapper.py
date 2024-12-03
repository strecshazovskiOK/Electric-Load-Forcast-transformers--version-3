from __future__ import annotations
import os
from pathlib import Path
from typing import Union, Any, Dict, List, Optional, Tuple, Type, cast
from contextlib import nullcontext
import torch
from torch import nn
from os import PathLike
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD
from torch.optim.rmsprop import RMSprop
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast  # Updated import for autocast
from torch.amp.grad_scaler import GradScaler  # Updated import for GradScaler

from models.losses.custom_losses import MAPE
from training.base.base_trainer import TrainingEpoch
from training.reports.training_report import TrainingReport

from ..interfaces import WrapperInterface
from ..base.base_model import BaseModel
from ..registry.model_types import ModelType

from torch.optim.lr_scheduler import (
    OneCycleLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    _LRScheduler,
    LRScheduler
)

from utils.logging.logger import Logger

# Get module logger
logger = Logger.get_logger(__name__)

class PyTorchWrapper(WrapperInterface):
    """Wrapper for PyTorch models providing consistent training and inference interface."""

    def __init__(self, model: BaseModel, model_type: ModelType, config: Dict[str, Any]):
        self.model = model
        self.model_type = model_type
        self.config = config

        # Device handling with improved CUDA settings
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        if self.device.type == 'cuda':
            # Enable TF32 for better performance on Ampere GPUs (like RTX 4060)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Enable cuDNN autotuner
            torch.backends.cudnn.benchmark = True

        logger.info(f"Using device: {self.device}")
        logger.debug(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logger.debug(f"CUDA device count: {torch.cuda.device_count()}")
            logger.debug(f"Current CUDA device: {torch.cuda.current_device()}")

        # Update mixed precision settings
        self.use_amp = config.get('use_mixed_precision', True) and self.device.type == 'cuda'
        self.dtype = torch.float32  # Base dtype for model parameters
        self.compute_dtype = torch.float16 if self.use_amp else torch.float32  # dtype for computations
        
        # Move model to device with correct dtype
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        
        # Initialize grad scaler exactly once
        self.grad_scaler = GradScaler(enabled=True) if self.use_amp else None
            
        # Set default tensor type for the model
        torch.set_default_dtype(self.dtype)

        # Move model to device
        self.model = self.model.to(self.device)
        logger.debug(f"Model device after moving: {next(self.model.parameters()).device}")

        # Training configuration
        self.batch_size = config.get('batch_size', 128)  # Increased for RTX 4060
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.max_epochs = config.get('max_epochs', 100)
        self.gradient_clip_val = config.get('gradient_clip_val', 1.0)
        self.accumulation_steps = config.get('accumulation_steps', 4)

        # Setup optimizer and criterion
        self.optimizer = self._setup_optimizer()
        self.criterion = self._setup_criterion()

        # Memory management settings
        if self.device.type == 'cuda':
            # Set memory allocator settings for RTX 4060 (8GB VRAM)
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available VRAM
            self.empty_cache_frequency = config.get('empty_cache_frequency', 100)
            self.max_memory_allocated = 0.0
            
        

    def _setup_data_loader(self, dataset: TorchDataset, shuffle: bool = True) -> DataLoader:
        """Create an optimized DataLoader for GPU training."""
        cpu_count = os.cpu_count() or 4
        num_workers = min(4, cpu_count // 2)  # Optimized for 32GB RAM

        data_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 2,
        }

        if self.device.type == 'cuda':
            data_loader_kwargs['pin_memory_device'] = 'cuda'  # Only pass when using CUDA

        return DataLoader(dataset, **data_loader_kwargs)
        
    def _setup_scheduler(self, train_loader: DataLoader) -> Optional[Union[LRScheduler, ReduceLROnPlateau]]:
        """Initialize learning rate scheduler with improved defaults."""
        scheduler_name = self.config.get('scheduler', 'one_cycle').lower()
        scheduler_config = self.config.get('scheduler_config', {})

        if scheduler_name == 'one_cycle':
            total_steps = self.max_epochs * len(train_loader)
            steps_per_epoch = len(train_loader)

            return cast(LRScheduler, OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                epochs=self.max_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=scheduler_config.get('pct_start', 0.3),
                div_factor=scheduler_config.get('div_factor', 25.0),
                final_div_factor=scheduler_config.get('final_div_factor', 1000.0),
                anneal_strategy=scheduler_config.get('anneal_strategy', 'cos')
            ))

        elif scheduler_name == 'cosine':
            return cast(LRScheduler, CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_epochs,
                eta_min=scheduler_config.get('eta_min', 1e-6)
            ))

        elif scheduler_name == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )

        elif scheduler_name == 'cosineannealinglr':
            return cast(LRScheduler, CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_epochs,
                eta_min=scheduler_config.get('eta_min', 1e-6)
            ))

        elif scheduler_name == 'steplr':
            return cast(LRScheduler, lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            ))

        else:
            logger.warning(f"Unknown scheduler: {scheduler_name}. No scheduler will be used.")
            return None

    def _setup_optimizer(self) -> Optimizer:
        """Initialize optimizer with improved defaults."""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        optimizer_config = self.config.get('optimizer_config', {})

        # Base parameters all optimizers support
        base_params = {
            'lr': self.learning_rate,
            'weight_decay': optimizer_config.get('weight_decay', 0.01)  # Updated default
        }

        # Create optimizer with appropriate parameters
        if optimizer_name == 'sgd':
            sgd_params = {
                'momentum': optimizer_config.get('momentum', 0.9),
                'dampening': optimizer_config.get('dampening', 0),
                'nesterov': optimizer_config.get('nesterov', True)  # Enable Nesterov by default
            }
            return SGD(self.model.parameters(), **base_params, **sgd_params)

        elif optimizer_name == 'adam':
            adam_params = {
                'betas': optimizer_config.get('betas', (0.9, 0.98)),  # Updated for transformer
                'eps': optimizer_config.get('eps', 1e-8),
                'amsgrad': optimizer_config.get('amsgrad', False)  # Typically False
            }
            return Adam(self.model.parameters(), **base_params, **adam_params)

        elif optimizer_name == 'adamw':
            adamw_params = {
                'betas': optimizer_config.get('betas', (0.9, 0.98)),  # Updated for transformer
                'eps': optimizer_config.get('eps', 1e-8),
                'amsgrad': optimizer_config.get('amsgrad', False)  # Typically False
            }
            return AdamW(self.model.parameters(), **base_params, **adamw_params)

        elif optimizer_name == 'rmsprop':
            rmsprop_params = {
                'alpha': optimizer_config.get('alpha', 0.99),
                'eps': optimizer_config.get('eps', 1e-8),
                'momentum': optimizer_config.get('momentum', 0.9),
                'centered': optimizer_config.get('centered', False)
            }
            return RMSprop(self.model.parameters(), **base_params, **rmsprop_params)

        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _setup_criterion(self) -> nn.Module:
        """Initialize loss function based on config."""
        criterion_name = self.config.get('criterion', 'mse').lower()
        criterion_config = self.config.get('criterion_config', {})

        criteria: Dict[str, nn.Module] = {
            'mse': nn.MSELoss(**criterion_config),
            'mae': nn.L1Loss(**criterion_config),
            'mape': MAPE(**criterion_config)
        }

        if criterion_name not in criteria:
            raise ValueError(f"Unknown criterion: {criterion_name}")

        return criteria[criterion_name]

    def train(self, train_dataset: TorchDataset, validation_dataset: Optional[TorchDataset] = None) -> TrainingReport:
        """Train the model with modern mixed precision and memory optimization."""
        logger.info("Starting training")
        
        # Clear GPU cache before training
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.debug(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        self.train_loader = self._setup_data_loader(train_dataset, shuffle=True)
        val_loader = self._setup_data_loader(validation_dataset, shuffle=False) if validation_dataset else None

        # Initialize scheduler
        self.scheduler = self._setup_scheduler(self.train_loader)

        # Training state
        train_losses = []
        val_losses = []
        learning_rates = []
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = self.config.get('early_stopping_patience', 10)

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                try:
                    # Move data to device efficiently
                    if isinstance(data, tuple):
                        data = tuple(d.to(self.device, non_blocking=True) for d in data)
                    else:
                        data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

                    # Forward pass with automatic mixed precision
                    with autocast('cuda') if self.use_amp else nullcontext():
                        outputs = self.model(data)
                        loss = self.criterion(outputs, target)
                        loss = loss / self.accumulation_steps

                    # Backward pass with gradient scaling
                    if self.use_amp and self.grad_scaler is not None:
                        self.grad_scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Gradient accumulation step
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        if self.gradient_clip_val > 0:
                            if self.use_amp and self.grad_scaler is not None:
                                self.grad_scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

                        # Optimizer step with modern AMP
                        if self.use_amp and self.grad_scaler is not None:
                            self.grad_scaler.step(self.optimizer)
                            self.grad_scaler.update()
                        else:
                            self.optimizer.step()

                        self.optimizer.zero_grad(set_to_none=True)

                    # Memory management for RTX 4060
                    if self.device.type == 'cuda' and batch_idx % self.empty_cache_frequency == 0:
                        current_memory = torch.cuda.memory_allocated() / 1024**3
                        self.max_memory_allocated = max(self.max_memory_allocated, current_memory)
                        
                        if current_memory > 7.0:  # Conservative threshold for 8GB VRAM
                            torch.cuda.empty_cache()

                    epoch_loss += loss.item() * self.accumulation_steps
                    num_batches += 1

                except RuntimeError as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                    continue

            # Calculate average loss and update metrics
            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            train_losses.append(avg_train_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            # Validation phase
            if val_loader:
                val_loss = self._validate(val_loader)
                val_losses.append(val_loss)

                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save('best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after epoch {epoch}")
                        break

            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loader else avg_train_loss)
                else:
                    self.scheduler.step()

            # Log progress
            logger.info(
                f"Epoch {epoch}/{self.max_epochs} - "
                f"Train Loss: {avg_train_loss:.6f}"
                + (f", Val Loss: {val_loss:.6f}" if val_loader else "")
                + f" - LR: {current_lr:.6f}"
            )

        return TrainingReport(
            train_losses=train_losses,
            val_losses=val_losses,
            learning_rates=learning_rates,
            epochs=epoch,
            additional_metrics={'best_val_loss': [best_val_loss]}  # Updated to use list
        )
        
        
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data, target in val_loader:
                try:
                    # Move data to device
                    if isinstance(data, tuple):
                        data = tuple(d.to(self.device, non_blocking=True) for d in data)
                    else:
                        data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

                    # Forward pass with autocast if using mixed precision
                    with autocast('cuda') if self.use_amp else nullcontext():
                        outputs = self.model(data)
                        loss = self.criterion(outputs, target)

                    val_loss += loss.item()
                    num_batches += 1

                except RuntimeError as e:
                    logger.error(f"Runtime error during validation: {str(e)}")
                    torch.cuda.empty_cache()
                    continue  # Skip this batch

        avg_val_loss = val_loss / num_batches if num_batches > 0 else float('inf')
        return avg_val_loss

    def predict(self, dataset: TorchDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions using the model."""
        data_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            num_workers=min(8, (os.cpu_count() or 4) // 2),
            pin_memory=torch.cuda.is_available(),
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=4
        )

        self.model.eval()
        predictions: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []

        with torch.no_grad():
            for data, target in data_loader:
                try:
                    # Move data to device
                    if isinstance(data, tuple):
                        data = tuple(d.to(self.device, non_blocking=True) for d in data)
                    else:
                        data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

                    # Forward pass with autocast if using mixed precision
                    with autocast('cuda') if self.use_amp else nullcontext():
                        output = self.model(data)

                    predictions.append(output.cpu())
                    targets.append(target.cpu())

                except RuntimeError as e:
                    logger.error(f"Runtime error during prediction: {str(e)}")
                    torch.cuda.empty_cache()
                    continue  # Skip this batch

        return torch.cat(predictions), torch.cat(targets)

    def save(self, path: Union[str, Path]) -> None:
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'grad_scaler_state_dict': self.grad_scaler.state_dict() if self.grad_scaler else None,
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.grad_scaler and checkpoint.get('grad_scaler_state_dict'):
            self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])
        logger.info(f"Model loaded from {path}")

    def training_step(
        self,
        batch_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],  # Changed type hint
        batch_target: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> float:
        """Process a single training batch with optimized GPU handling."""
        self.optimizer.zero_grad(set_to_none=True)
        
        try:
            # Convert inputs to correct dtype (always float32 for parameters)
            if isinstance(batch_input, tuple):
                batch_input = tuple(b.to(device=self.device, dtype=self.dtype) for b in batch_input)
                if len(batch_input) != 2:
                    raise ValueError(f"Expected tuple of 2 tensors but got {len(batch_input)}")
                batch_input = cast(Tuple[torch.Tensor, torch.Tensor], batch_input)  # Add explicit cast
            else:
                batch_input = batch_input.to(device=self.device, dtype=self.dtype)
            
            batch_target = batch_target.to(device=self.device, dtype=self.dtype)
            
            # Handle transformer models
            if self.model_type.is_transformer and isinstance(batch_input, tuple):
                src, tgt = batch_input
                src_mask = (src_mask.to(device=self.device, dtype=self.dtype) 
                           if src_mask is not None 
                           else self.model.generate_square_subsequent_mask(src.size(1)).to(device=self.device, dtype=self.dtype))
                tgt_mask = (tgt_mask.to(device=self.device, dtype=self.dtype)
                           if tgt_mask is not None
                           else self.model.generate_square_subsequent_mask(tgt.size(1)).to(device=self.device, dtype=self.dtype))
                
                with autocast(device_type='cuda', dtype=self.compute_dtype) if self.use_amp else nullcontext():
                    output = self.model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
                    loss = self.criterion(output, batch_target)
            else:
                if isinstance(batch_input, tuple):
                    batch_input = batch_input[0]
                with autocast(device_type='cuda', dtype=self.compute_dtype) if self.use_amp else nullcontext():
                    output = self.model(batch_input)
                    loss = self.criterion(output, batch_target)
            
            # Modified gradient handling with proper checks
            if self.use_amp and self.grad_scaler is not None:
                assert self.grad_scaler is not None  # For type checker
                self.grad_scaler.scale(loss).backward()
                
                if self.gradient_clip_val > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()
            
            return loss.item()
                
        except RuntimeError as e:
            logger.error(f"Training step failed: {str(e)}")
            logger.debug("Device and dtype mapping:")
            logger.debug(f"Model: device={next(self.model.parameters()).device}, dtype={next(self.model.parameters()).dtype}")
            if isinstance(batch_input, tuple):
                logger.debug(f"Input tensors: {[(b.device, b.dtype) for b in batch_input]}")
            else:
                logger.debug(f"Input tensor: device={batch_input.device}, dtype={batch_input.dtype}")
            logger.debug(f"Target tensor: device={batch_target.device}, dtype={batch_target.dtype}")
            raise

    def validation_step(
        self,
        batch_input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        batch_target: torch.Tensor,
        **kwargs: Any
    ) -> float:
        """Process a single validation batch.
        
        Args:
            batch_input: Input tensor or tuple of tensors
            batch_target: Target tensor
            **kwargs: Additional arguments like masks for transformers
            
        Returns:
            float: Loss value for this batch
        """
        try:
            # Handle transformer models
            if self.model_type.is_transformer and isinstance(batch_input, tuple):
                src, tgt = batch_input
                # Fix mask handling - don't use get() directly on kwargs
                src_mask = kwargs.get('src_mask')
                tgt_mask = kwargs.get('tgt_mask')
                
                if src_mask is None:
                    src_mask = self.model.generate_square_subsequent_mask(src.size(1)).to(self.device)
                if tgt_mask is None:
                    tgt_mask = self.model.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
                
                with torch.no_grad():
                    with autocast('cuda') if self.use_amp else nullcontext():
                        output = self.model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            else:
                if isinstance(batch_input, tuple):
                    batch_input = batch_input[0]
                with torch.no_grad():
                    with autocast('cuda') if self.use_amp else nullcontext():
                        output = self.model(batch_input)
                
                loss = self.criterion(output, batch_target)

            loss = self.criterion(output, batch_target)
            return loss.item()
                
        except RuntimeError as e:
            logger.error(f"Validation step failed: {str(e)}")
            logger.debug("Device mapping:")
            logger.debug(f"Model: {next(self.model.parameters()).device}")
            if isinstance(batch_input, tuple):
                logger.debug(f"Input tensors: {[b.device for b in batch_input]}")
            else:
                logger.debug(f"Input tensor: {batch_input.device}")
            logger.debug(f"Target tensor: {batch_target.device}")
            raise
