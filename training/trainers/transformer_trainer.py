# training/trainers/transformer_trainer.py
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from typing import Tuple

from training.callbacks.model_checkpoint import ModelCheckpoint
from training.reports.training_report import TrainingReport

from ..base.base_trainer import BaseTrainer

def create_mask(size: int) -> torch.Tensor:
    # sourcery skip: remove-unnecessary-cast
    """Create attention mask for transformer."""
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class TransformerTrainer(BaseTrainer):
    """Trainer implementation for transformer models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_callback = ModelCheckpoint(
            filepath='checkpoints/transformer-{epoch:02d}-{val_loss:.2f}.pt',
            monitor='val_loss',
            save_best_only=True
        )
        self.checkpoint_callback.on_training_begin(self.model, vars(self.args))

    def execute_model_on_batch(
            self,
            encoder_input: torch.Tensor,
            decoder_input: torch.Tensor,
            device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute model using generative approach."""
        batch_size = encoder_input.shape[0]
        decoder_sequence_length = self.args.transformer_labels_count + self.args.forecasting_horizon

        # Prepare decoder input
        expected = decoder_input[:, :decoder_sequence_length, 0]
        u = decoder_input[:, :decoder_sequence_length, 1:]
        o1 = decoder_input[:, :self.args.transformer_labels_count, 0:1]
        o2 = torch.zeros([batch_size, self.args.forecasting_horizon, 1]).to(device)
        adjusted_decoder_input = torch.cat([torch.cat([o1, o2], dim=1), u], dim=2).to(device)

        # Create mask and forward pass
        target_mask = create_mask(decoder_sequence_length).to(device)
        predicted = self.model(encoder_input, adjusted_decoder_input, tgt_mask=target_mask)
        predicted = torch.reshape(predicted, torch.Size([batch_size, decoder_sequence_length]))

        return predicted, expected

    def execute_model_one_step_ahead(
            self,
            encoder_input: torch.Tensor,
            decoder_input: torch.Tensor,
            device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute model using one-step-ahead approach."""
        batch_size = encoder_input.shape[0]
        expected = decoder_input[:, 1:self.args.transformer_labels_count + 1, 0]

        # Prepare decoder input
        u = decoder_input[:, 1:-self.args.forecasting_horizon + 1, 1:]
        o1 = decoder_input[:, :self.args.transformer_labels_count, 0:1]
        adjusted_decoder_input = torch.cat([o1, u], dim=2).to(device)

        # Create mask and forward pass
        target_mask = create_mask(self.args.transformer_labels_count).to(device)
        predicted = self.model(encoder_input, adjusted_decoder_input, tgt_mask=target_mask)
        predicted = torch.reshape(predicted, torch.Size([batch_size, self.args.transformer_labels_count]))

        return predicted, expected

    def train_phase(self, device: str) -> float:
        self.model.train()
        total_training_loss = 0.0

        for encoder_input, decoder_input in self.train_data_loader:
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)

            self.optimizer.zero_grad()

            if self.args.transformer_use_teacher_forcing:
                expected = decoder_input[:, self.args.transformer_labels_count:, 0].detach().clone()
                decoder_input[:, 1:, 0:1] = decoder_input[:, :-1, 0:1]

                target_mask = create_mask(decoder_input.shape[1]).to(device)
                predicted = self.model(encoder_input, decoder_input, tgt_mask=target_mask)
                predicted = torch.reshape(
                    predicted,
                    torch.Size([encoder_input.shape[0], self.args.transformer_labels_count + self.args.forecasting_horizon])
                )
                predicted = predicted[:, self.args.transformer_labels_count:]

            elif self.args.transformer_use_auto_regression:
                predicted, expected = self.execute_model_one_step_ahead(encoder_input, decoder_input, device)
                predicted = predicted[:, self.args.transformer_labels_count - 1:]
                expected = expected[:, self.args.transformer_labels_count - 1:]

            else:  # generative approach
                predicted, expected = self.execute_model_on_batch(encoder_input, decoder_input, device)

            training_loss = self.loss_criterion(predicted, expected)
            training_loss.backward()
            self.optimizer.step()
            total_training_loss += training_loss.item()

        return total_training_loss / len(self.train_data_loader)

    def validation_phase(self, device: str) -> float:
        self.model.eval()
        total_validation_loss = 0.0

        with torch.no_grad():
            for encoder_input, decoder_input in self.validation_data_loader:
                encoder_input = encoder_input.to(device)
                decoder_input = decoder_input.to(device)

                if self.args.transformer_use_teacher_forcing:
                    predicted, expected = self._teacher_forcing_validation(encoder_input, decoder_input, device)
                elif self.args.transformer_use_auto_regression:
                    predicted, expected = self.execute_model_one_step_ahead(encoder_input, decoder_input, device)
                    predicted = predicted[:, self.args.transformer_labels_count - 1:]
                    expected = expected[:, self.args.transformer_labels_count - 1:]
                else:  # generative approach
                    predicted, expected = self.execute_model_on_batch(encoder_input, decoder_input, device)
                    predicted = predicted[:, self.args.transformer_labels_count:]
                    expected = expected[:, self.args.transformer_labels_count:]

                validation_loss = self.loss_criterion(predicted, expected)
                total_validation_loss += validation_loss.item()

        return total_validation_loss / len(self.validation_data_loader)

    def _teacher_forcing_validation(
            self,
            encoder_input: torch.Tensor,
            decoder_input: torch.Tensor,
            device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validation step for teacher forcing approach."""
        expected = decoder_input[:, self.args.transformer_labels_count:, 0].detach().clone().to(device)
        decoder_input[:, 1:, 0] = decoder_input[:, :-1, 0]

        start_decoder_input = decoder_input[:, :self.args.transformer_labels_count + 1, :].to(device)

        for i in range(1, 1 + self.args.forecasting_horizon):
            target_mask = create_mask(start_decoder_input.shape[1]).to(device)
            predicted = self.model(encoder_input, start_decoder_input, tgt_mask=target_mask).to(device)

            if i == self.args.forecasting_horizon:
                known_decoder_input = torch.zeros(
                    start_decoder_input.shape[0], 1, start_decoder_input.shape[2] - 1
                ).to(device)
            else:
                known_decoder_input = decoder_input[
                                    :,
                                    self.args.transformer_labels_count + i:self.args.transformer_labels_count + i + 1,
                                    1:
                                    ].to(device)
            new_predicted = predicted[
                            :,
                            self.args.transformer_labels_count + i - 1:self.args.transformer_labels_count + i,
                            0:1
                            ].to(device)

            predicted = torch.cat([new_predicted, known_decoder_input], dim=2).to(device)
            start_decoder_input = torch.cat([start_decoder_input, predicted], dim=1).to(device)

        predicted = start_decoder_input[:, self.args.transformer_labels_count + 1:, 0].to(device)
        return predicted, expected

    def train(self) -> TrainingReport:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Used device: ', device)
        self.model = self.model.to(device)

        train_losses = []
        val_losses = []
        learning_rates = []
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

            # Save checkpoint
            self.checkpoint_callback.on_epoch_end(epoch, {
                'val_loss': validation_loss,
                'train_loss': training_loss
            })

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

            print(f'Epoch {epoch}: training_loss={training_loss:.4f}, validation_loss={validation_loss:.4f}')

        # Load best model and move to CPU
        self.model.load_state_dict(self.best_model_state)
        self.model = self.model.to('cpu')

        return TrainingReport(
            train_losses=train_losses,
            val_losses=val_losses,
            learning_rates=learning_rates,
            epochs=self.epochs_count,
            early_stopping_epoch=epoch if epochs_without_validation_loss_decrease > self.args.early_stopping_patience else None
        )