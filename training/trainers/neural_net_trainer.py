# training/trainers/neural_net_trainer.py
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch

from ..base.base_trainer import BaseTrainer

class NeuralNetTrainer(BaseTrainer):
    """Trainer implementation for neural networks."""

    def __init__(
            self,
            train_data_loader: DataLoader,
            validation_data_loader: DataLoader,
            model: nn.Module,
            loss_criterion,
            optimizer,
            epochs_count: int,
            learning_rate_scheduler: StepLR,
            args
    ):
        super().__init__(
            train_data_loader,
            validation_data_loader,
            model,
            loss_criterion,
            optimizer,
            epochs_count,
            learning_rate_scheduler,
            args
        )
        self.callbacks = []

    def train_phase(self, device: str) -> float:
        self.model.train()
        total_training_loss = 0.0

        for inputs, targets in self.train_data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            self.optimizer.zero_grad()

            output = self.model(inputs)
            training_loss = self.loss_criterion(output, targets)
            training_loss.backward()
            self.optimizer.step()

            total_training_loss += training_loss.item()

        return total_training_loss / len(self.train_data_loader)

    def validation_phase(self, device: str) -> float:
        self.model.eval()
        total_validation_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.validation_data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                output = self.model(inputs)
                output = output.to(device)

                validation_loss = self.loss_criterion(output, targets)
                total_validation_loss += validation_loss.item()

        return total_validation_loss / len(self.validation_data_loader)