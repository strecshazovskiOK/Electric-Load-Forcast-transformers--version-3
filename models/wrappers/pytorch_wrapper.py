# models/wrappers/pytorch_wrapper.py
from __future__ import annotations
from pathlib import Path
from typing import Union
from typing import Any, Dict, List, Optional, Tuple, Union, Type, cast
import torch
from torch import nn
from os import PathLike
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset as TorchDataset  # Rename to avoid conflict
from torch.utils.data import DataLoader

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
        
        # Ensure consistent device handling
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        logger.info(f"Using device: {self.device}")
        logger.debug(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.debug(f"CUDA device count: {torch.cuda.device_count()}")
            logger.debug(f"Current CUDA device: {torch.cuda.current_device()}")
        
        # Move model to device and verify
        self.model = self.model.to(self.device)
        logger.debug(f"Model device after moving: {next(self.model.parameters()).device}")

        # Training configuration
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.max_epochs = config.get('max_epochs', 100)
        self.gradient_clip_val = config.get('gradient_clipping', 1.0)

        # Setup optimizer and criterion (these will inherit the model's device)
        self.optimizer = self._setup_optimizer()
        self.criterion = self._setup_criterion()
        self.scheduler = None
        self.debug_counter = 0
        self.max_debug_prints = 2
        

    def _setup_scheduler(self) -> Optional[Union[LRScheduler, ReduceLROnPlateau]]:
        """Initialize learning rate scheduler with improved defaults."""
        scheduler_name = self.config.get('scheduler', 'one_cycle')
        scheduler_config = self.config.get('scheduler_config', {})

        if scheduler_name == 'one_cycle':
            total_steps = self.max_epochs * len(self.train_loader)
            steps_per_epoch = len(self.train_loader)
            
            return cast(LRScheduler, OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                epochs=self.max_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=scheduler_config.get('pct_start', 0.3),
                div_factor=scheduler_config.get('div_factor', 25.0),
                final_div_factor=scheduler_config.get('final_div_factor', 10000.0),
                anneal_strategy='cos'
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
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        
        return None

    def _setup_optimizer(self) -> Optimizer:
        """Initialize optimizer with improved defaults."""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        optimizer_config = self.config.get('optimizer_config', {})
        
        # Base parameters all optimizers support
        base_params = {
            'lr': self.learning_rate,
            'weight_decay': optimizer_config.get('weight_decay', 0.05)  # Updated default
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
                'amsgrad': optimizer_config.get('amsgrad', True)  # Enable AMSGrad by default
            }
            return Adam(self.model.parameters(), **base_params, **adam_params)
            
        elif optimizer_name == 'adamw':
            adamw_params = {
                'betas': optimizer_config.get('betas', (0.9, 0.98)),  # Updated for transformer
                'eps': optimizer_config.get('eps', 1e-8),
                'amsgrad': optimizer_config.get('amsgrad', True)  # Enable AMSGrad by default
            }
            return AdamW(self.model.parameters(), **base_params, **adamw_params)
            
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")


    def train(
            self,
            train_dataset: TorchDataset,
            validation_dataset: Optional[TorchDataset] = None
    ) -> TrainingReport:
        """Train the model with proper device handling."""
        
        logger.info("Starting training")
        logger.debug(f"Training device setup: CUDA available: {torch.cuda.is_available()}, Using device: {self.device}")
        
        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            
        # Ensure model is on correct device
        self.model = self.model.to(self.device)
        
        # Setup data loading with device pinning
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=512,  # Increased batch size from 32 to 256
            shuffle=True,
            num_workers=4,  # Keep other parameters same for now
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=2
        )

        val_loader = None
        if validation_dataset is not None:
            val_loader = DataLoader(
                validation_dataset,
                batch_size=512,  # Match training batch size
                num_workers=4,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True,
                prefetch_factor=2
            )


        # Initialize scheduler
        self.scheduler = self._setup_scheduler()

        # Training loop variables
        train_losses = []
        val_losses = []
        learning_rates = []
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = self.config.get('early_stopping_patience', 10)
        
        # Optional: Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                try:
                    # Non-blocking transfer to GPU
                    if isinstance(data, tuple):
                        data = tuple(d.to(self.device, non_blocking=True) for d in data)
                    else:
                        data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    
                    loss = self.training_step(data, target)
                    train_loss += loss
                    num_batches += 1

                    if (batch_idx + 1) % 50 == 0:
                        logger.info(f"Batch [{batch_idx+1}/{len(self.train_loader)}] - Loss: {loss:.4f}")
                        if torch.cuda.is_available():
                            logger.debug(f"GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

                except RuntimeError as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue


            # Calculate average training loss
            if num_batches > 0:
                train_loss /= num_batches
                train_losses.append(train_loss)
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                learning_rates.append(current_lr)

                logger.info(f"Epoch {epoch+1}/{self.max_epochs}: Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")


            # Validation phase
            if val_loader:
                val_loss = self._validate(val_loader)
                val_losses.append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pt')
                else:
                    patience_counter += 1

                # Step schedulers that are not OneCycleLR
                if self.scheduler is not None and not isinstance(self.scheduler, OneCycleLR):
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.warning(f"Early stopping triggered after {epoch+1} epochs")
                    # Load best model
                    self.model.load_state_dict(torch.load('best_model.pt'))
                    break
            else:
                logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")

        # Create additional_metrics dictionary (renamed from training_metrics)
        additional_metrics = {
            'best_val_loss': best_val_loss if val_loader else None,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'final_learning_rate': learning_rates[-1] if learning_rates else None
        }

        metrics = {  # Add metrics dictionary
            'best_val_loss': best_val_loss if val_loader else None,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'final_learning_rate': learning_rates[-1] if learning_rates else None
        }

        return TrainingReport(
            train_losses=train_losses,
            val_losses=val_losses,
            learning_rates=learning_rates,
            epochs=epoch + 1,
            additional_metrics=additional_metrics  # Use additional_metrics instead of metrics
        )
        
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

    


    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()

        return val_loss / len(val_loader)

    def predict(self, dataset: TorchDataset) -> tuple[torch.Tensor, torch.Tensor]:
        """Make predictions using the model."""
        data_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )

        self.model.eval()
        predictions: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []

        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                predictions.append(self.model(data).cpu())
                targets.append(target)

        return torch.cat(predictions), torch.cat(targets)

    def save(self, path: Union[str, Path]) -> None:
        """Save model state."""
        torch.save({ # type: ignore
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config
        }, path)

    def load(self, path: Union[str, Union[str, Path]]) -> None:
        """Load model state."""
        checkpoint = torch.load(path) # type: ignore
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def training_step(
        self,
        batch_input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        batch_target: torch.Tensor,
        **kwargs
    ) -> float:
        """Process a single training batch with optimized GPU handling."""
        self.optimizer.zero_grad()
        
        try:
            # Handle transformer models
            if self.model_type.is_transformer and isinstance(batch_input, tuple):
                src, tgt = batch_input
                # Use masks from kwargs if provided, else generate
                src_mask = kwargs.get('src_mask', self.model.generate_square_subsequent_mask(src.size(1)).to(self.device))
                tgt_mask = kwargs.get('tgt_mask', self.model.generate_square_subsequent_mask(tgt.size(1)).to(self.device))
                
                output = self.model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            else:
                if isinstance(batch_input, tuple):
                    batch_input = batch_input[0]
                output = self.model(batch_input)
            
            loss = self.criterion(output, batch_target)
            loss.backward()
            
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
            self.optimizer.step()
            
            return loss.item()
                
        except RuntimeError as e:
            logger.error(f"Training step failed: {str(e)}")
            logger.debug("Device mapping:")
            logger.debug(f"Model: {next(self.model.parameters()).device}")
            if isinstance(batch_input, tuple):
                logger.debug(f"Input tensors: {[b.device for b in batch_input]}")
            else:
                logger.debug(f"Input tensor: {batch_input.device}")
            logger.debug(f"Target tensor: {batch_target.device}")
            raise
    
        
    def validation_step(
        self, 
        batch_input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
        batch_target: torch.Tensor
    ) -> float:
        """Perform single validation step with proper device handling."""
        self.model.eval()
        
        try:
            # Move batch_target to device
            batch_target = batch_target.to(self.device)
            
            with torch.no_grad():
                # Handle transformer models
                if self.model_type.is_transformer and isinstance(batch_input, tuple):
                    src, tgt = batch_input
                    src = src.to(self.device)
                    tgt = tgt.to(self.device)
                    
                    # Generate masks and move them to device
                    src_mask = self.model.generate_square_subsequent_mask(src.size(1)).to(self.device)
                    tgt_mask = self.model.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
                    
                    output = self.model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
                else:
                    # Ensure batch_input is a tensor and move to device
                    if isinstance(batch_input, tuple):
                        batch_input = batch_input[0]
                    batch_input = batch_input.to(self.device)
                    output = self.model(batch_input)
                
                loss = self.criterion(output, batch_target)
            
            return loss.item()
            
        except RuntimeError as e:
            logger.error(f"Error in validation step: {str(e)}")
            logger.error(f"Current devices:")
            logger.error(f"Model: {next(self.model.parameters()).device}")
            logger.error(f"Target: {batch_target.device}")
            if isinstance(batch_input, tuple):
                logger.error(f"Source: {batch_input[0].device}")
                logger.error(f"Target input: {batch_input[1].device}")
            else:
                logger.error(f"Input: {batch_input.device}")
            raise