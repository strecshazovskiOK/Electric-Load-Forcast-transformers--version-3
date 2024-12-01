# training/base/base_trainer.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from training.reports.training_report import TrainingReport  # Updated import path
import os


class TrainingEpoch:
    def __init__(self, epoch_number: int, training_loss: float, validation_loss: float):
        self.epoch_number = epoch_number
        self.training_loss = training_loss
        self.validation_loss = validation_loss
        self.callbacks = []

    def serialize(self) -> Dict[str, Any]:
        return {
            'epochNumber': self.epoch_number,
            'trainingLoss': self.training_loss,
            'validationLoss': self.validation_loss
        }


class BaseTrainer(ABC):
    def __init__(
            self,
            train_data_loader: DataLoader[torch.Tensor],
            validation_data_loader: DataLoader[torch.Tensor],
            model: nn.Module,
            loss_criterion: nn.Module,
            optimizer: Optimizer,
            epochs_count: int,
            learning_rate_scheduler: StepLR,
            args: Any
    ):
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.model = model
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.epochs_count = epochs_count
        self.learning_rate_scheduler = learning_rate_scheduler
        self.args = args
        self.best_model_state = {}
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self) -> TrainingReport:  # Fix return type annotation
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Used device: ', device)
        self.model = self.model.to(device)

        train_losses: List[float] = []
        val_losses: List[float] = []
        learning_rates: List[float] = []
        epochs_without_validation_loss_decrease = 0
        minimum_average_validation_loss = float('inf')

        for epoch in range(self.epochs_count):
            # Training phase
            training_loss = self.train_phase(device)
            train_losses.append(training_loss)

            # Validation phase
            validation_loss = self.validation_phase(device)
            val_losses.append(validation_loss)
            learning_rates.append(self.optimizer.param_groups[0]['lr'])

            self.learning_rate_scheduler.step()

            if self.args.use_early_stopping:
                if minimum_average_validation_loss <= validation_loss:
                    epochs_without_validation_loss_decrease += 1
                else:
                    epochs_without_validation_loss_decrease = 0
                    minimum_average_validation_loss = validation_loss
                    self.best_model_state = self.model.state_dict().copy()

                if epochs_without_validation_loss_decrease > self.args.early_stopping_patience:
                    print('Early stopping has happened at epoch', epoch)
                    break

            # Save best model state
            if validation_loss < minimum_average_validation_loss:
                self.best_model_state = self.model.state_dict().copy()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'validation_loss': validation_loss,
                }, f'{self.checkpoint_dir}/best_model.pt')

            print('Epoch: ', epoch)
            print('Average training loss: ', training_loss)
            print('Average validation loss: ', validation_loss)

        # Load best model and move to CPU
        self.model.load_state_dict(self.best_model_state)
        self.model = self.model.to('cpu')

        return TrainingReport(  # Use renamed class
            train_losses=train_losses,
            val_losses=val_losses,
            learning_rates=learning_rates,
            epochs=self.epochs_count,
            early_stopping_epoch=epoch if epochs_without_validation_loss_decrease > self.args.early_stopping_patience else None
        )

    @abstractmethod
    def train_phase(self, device: str) -> float:
        """Execute training phase for one epoch."""
        pass

    @abstractmethod
    def validation_phase(self, device: str) -> float:
        """Execute validation phase for one epoch."""
        pass