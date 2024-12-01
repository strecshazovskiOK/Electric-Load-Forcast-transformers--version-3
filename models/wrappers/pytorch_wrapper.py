# models/wrappers/pytorch_wrapper.py
from __future__ import annotations
from pathlib import Path
from typing import Union
from typing import Union
from typing import Any, Dict, List, Optional, Tuple, Union, Type
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


class PyTorchWrapper(WrapperInterface):
    """Wrapper for PyTorch models providing consistent training and inference interface."""

    def __init__(self, model: BaseModel, model_type: ModelType, config: Dict[str, Any]):
        self.model = model
        self.model_type = model_type
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Training configuration
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.max_epochs = config.get('max_epochs', 100)
        self.gradient_clip_val = config.get('gradient_clipping', 1.0)

        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = self._setup_criterion()

    def _setup_optimizer(self) -> Optimizer:
        """Initialize optimizer with improved defaults."""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        optimizer_config = self.config.get('optimizer_config', {})
        
        # Base parameters all optimizers support
        base_params = {
            'lr': self.learning_rate,
            'weight_decay': optimizer_config.get('weight_decay', 0.01)
        }

        # Create optimizer with appropriate parameters
        if optimizer_name == 'sgd':
            sgd_params = {
                'momentum': optimizer_config.get('momentum', 0.9),
                'dampening': optimizer_config.get('dampening', 0),
                'nesterov': optimizer_config.get('nesterov', False)
            }
            return SGD(self.model.parameters(), **base_params, **sgd_params)
            
        elif optimizer_name == 'adam':
            adam_params = {
                'betas': optimizer_config.get('betas', (0.9, 0.999)),
                'eps': optimizer_config.get('eps', 1e-8),
                'amsgrad': optimizer_config.get('amsgrad', False)
            }
            return Adam(self.model.parameters(), **base_params, **adam_params)
            
        elif optimizer_name == 'adamw':
            adamw_params = {
                'betas': optimizer_config.get('betas', (0.9, 0.999)),
                'eps': optimizer_config.get('eps', 1e-8),
                'amsgrad': optimizer_config.get('amsgrad', False)
            }
            return AdamW(self.model.parameters(), **base_params, **adamw_params)
            
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")


    def _setup_scheduler(self) -> Optional[Union[lr_scheduler.StepLR, lr_scheduler.CosineAnnealingLR, lr_scheduler.ReduceLROnPlateau]]:
        """Initialize learning rate scheduler with improved defaults."""
        scheduler_name = self.config.get('scheduler', 'cosine')  # Default changed to cosine
        
        scheduler_config = self.config.get('scheduler_config', {
            'T_max': self.max_epochs,
            'eta_min': 1e-6
        })

        schedulers = {
            'step': optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            ),
            'cosine': optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                **scheduler_config
            ),
            'plateau': optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        }

        if scheduler_name not in schedulers:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        return schedulers[scheduler_name]


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

    def train(
            self,
            train_dataset: TorchDataset,
            validation_dataset: Optional[TorchDataset] = None
    ) -> TrainingReport:
        """Train the model with improved training loop."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = None
        if validation_dataset is not None:
            val_loader = DataLoader(
                validation_dataset,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True
            )

        train_losses = []
        val_losses = []
        learning_rates = []
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = self.config.get('early_stopping_patience', 15)

        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                try:
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    
                    # Gradient clipping
                    if self.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.gradient_clip_val
                        )
                    
                    self.optimizer.step()
                    train_loss += loss.item()

                    if (batch_idx + 1) % 50 == 0:
                        print(f"Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f}")

                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    raise e

            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

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

                # Update learning rate
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    # Load best model
                    self.model.load_state_dict(torch.load('best_model.pt'))
                    break
            else:
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")

        return TrainingReport(
            train_losses=train_losses,
            val_losses=val_losses,
            learning_rates=learning_rates,
            epochs=epoch + 1
        )


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
