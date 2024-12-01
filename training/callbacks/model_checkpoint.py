# training/callbacks/model_checkpoint.py
from typing import Dict, Any, Optional
import os
import torch
import torch.nn as nn

from .base_callback import TrainerCallback


class ModelCheckpoint(TrainerCallback):
    """Save model checkpoints during training."""

    def __init__(
            self,
            filepath: str,
            monitor: str = 'val_loss',
            save_best_only: bool = True,
            save_weights_only: bool = True
    ):
        self.model: Optional[nn.Module] = None
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.best_value = float('inf')

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        print(f"ModelCheckpoint: Epoch {epoch} ended.")
        current = logs.get(self.monitor)
        if current is None:
            print("No monitored value found in logs.")
            return

        # Convert epoch and float values to strings before formatting
        format_dict = {'epoch': epoch}
        for k, v in logs.items():
            format_dict[k] = v
    
        try:
            filepath = self.filepath.format(**format_dict)
            print(f"Formatted filepath: {filepath}")
        except Exception as e:
            print(f"Error formatting filepath: {e}")
            filepath = self.filepath.replace("{epoch}", f"{epoch:02d}")
            for k, v in logs.items():
                if isinstance(v, float):
                    filepath = filepath.replace(f"{{{k}:.2f}}", f"{v:.2f}")
                else:
                    filepath = filepath.replace(f"{{{k}}}", str(v))

        if self.save_best_only:
            if current < self.best_value:
                self.best_value = current
                if saved := self._save_model(filepath):
                    print(f"Model improved. Saved model to {filepath}")
                else:
                    print("Failed to save model")
            else:
                print("Model did not improve. Not saving.")
        elif saved := self._save_model(filepath):
            print(f"Model saved to {filepath}")
        else:
            print("Failed to save model")

    def _save_model(self, filepath: str) -> bool:
        """Save the model and return True if successful"""
        if self.model is None:
            print("Model is not initialized. Cannot save.")
            return False

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save model
            if self.save_weights_only:
                torch.save(self.model.state_dict(), filepath)
            else:
                torch.save(self.model, filepath)

            # Verify file exists after saving
            if os.path.exists(filepath):
                print(f"Model successfully saved to {filepath}")
                return True
            else:
                print(f"File not found after save attempt: {filepath}")
                return False

        except Exception as e:
            print(f"Failed to save model: {str(e)}")
            return False

    def on_training_begin(self, model: nn.Module, config: Dict[str, Any]) -> None:
        self.model = model

    def on_training_end(self, model: nn.Module, config: Dict[str, Any]) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        pass
