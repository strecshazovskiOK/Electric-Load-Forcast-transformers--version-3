import numpy as np
import pandas as pd
import torch
from typing import Tuple, List, Any

from data_loading.base.base_dataset import BaseDataset, DatasetConfig
from data_loading.features.time_features import CyclicalTimeFeature, WorkdayFeature
from data_loading.preprocessing.data_scaler import DataScaler
from data_loading.preprocessing.data_transformer import DataTransformer
from utils.logging.logger import Logger
from utils.logging.config import LoggerConfig, LogLevel

class TransformerDataset(BaseDataset):
    """Dataset implementation for transformer models with proper sequence handling"""

    def __init__(self, df: pd.DataFrame, config: DatasetConfig):
        super().__init__(df, config)
        self.rows: torch.Tensor = torch.empty(0)
        self.logger = Logger.get_logger(__name__)
        self._prepare_time_series_data()
        self._debug_counter = 0
        self._debug_frequency = 10000

    def __len__(self) -> int:
        """
        Calculate the number of available sequences in the dataset.
        
        Returns:
            Number of sequences that can be extracted from the data
        """
        # Make sure we have data
        if self.rows is None or len(self.rows) == 0:
            return 0
            
        # Calculate total sequence length needed for input and forecast
        total_window = (
            self.config.window_size +    # Input sequence length
            self.config.horizon_size     # Forecast sequence length
        )
        
        # Return maximum number of sequences we can create
        # Subtract total_window - 1 to ensure we have enough points for the last sequence
        return max(0, len(self.rows) - (total_window - 1))
    
    def _validate_gradients(self, tensors: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        """Validate gradient settings of output tensors."""
        (src_seq, tgt_input_seq), target_seq = tensors
        
        # Check gradient tracking
        if not src_seq.requires_grad:
            self.logger.warning("Source sequence missing gradient tracking")
            
        if not tgt_input_seq.requires_grad:
            self.logger.warning("Target input sequence missing gradient tracking")
            
        # We typically don't need gradients for the target
        # as it's used for loss computation
        
        # Validate tensor properties
        for name, tensor in [
            ("src_seq", src_seq), 
            ("tgt_input_seq", tgt_input_seq), 
            ("target_seq", target_seq)
        ]:
            if torch.isnan(tensor).any():
                raise ValueError(f"NaN values detected in {name}")
            if torch.isinf(tensor).any():
                raise ValueError(f"Inf values detected in {name}")

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Get a data item with proper gradient tracking."""
        if self.rows is None:
            raise ValueError("Dataset not properly initialized")

        input_window = self.config.window_size
        forecast_window = self.config.horizon_size

        try:
            # Get sequences with gradient tracking maintained
            src_seq = self.rows[index:index + input_window]
            full_target = self.rows[index + input_window:index + input_window + forecast_window]
            
            # Prepare sequences for teacher forcing while maintaining gradients
            tgt_input_seq = full_target[:-1]
            target_seq = full_target[1:, 0:1]  # Only keep energy consumption value

            # Debug information with gradient tracking status
            if self._debug_counter % self._debug_frequency == 0:
                self.logger.debug("Dataset sample info", {
                    "sample": self._debug_counter,
                    "source_shape": src_seq.shape,
                    "target_input_shape": tgt_input_seq.shape,
                    "target_shape": target_seq.shape,
                    "source_range": f"[{src_seq.min():.2f}, {src_seq.max():.2f}]",
                    "target_range": f"[{target_seq.min():.2f}, {target_seq.max():.2f}]",
                    "source_requires_grad": src_seq.requires_grad,
                    "target_input_requires_grad": tgt_input_seq.requires_grad,
                    "target_requires_grad": target_seq.requires_grad
                })
            
            self._debug_counter += 1

            # Add batch dimension while preserving gradients
            if len(src_seq.shape) == 2:
                src_seq = src_seq.unsqueeze(0)
                tgt_input_seq = tgt_input_seq.unsqueeze(0)
                target_seq = target_seq.unsqueeze(0)

            # Ensure tensors have proper gradient tracking
            src_seq.requires_grad_(True)
            tgt_input_seq.requires_grad_(True)
            # Note: target_seq typically doesn't need gradients as it's the ground truth

            return (src_seq, tgt_input_seq), target_seq
            
        except Exception as e:
            self.logger.error("Error in __getitem__", {
                "index": index,
                "error": str(e),
                "dataset_size": len(self.rows),
                "input_window": input_window,
                "forecast_window": forecast_window
            })
            raise


    def _prepare_time_series_data(self) -> None:
        """Prepare time series data with proper gradient tracking."""
        self.logger.info("Preparing transformer dataset", {
            "dataframe_shape": self._df.shape
        })
        
        if len(self._df) == 0:
            raise ValueError("Empty dataframe provided")

        try:
            # Initialize processors
            scaler = DataScaler(self.config.time_series_scaler)
            transformer = DataTransformer()

            # Process load data
            load_data = np.array(self._df[self.config.target_variable])
            time_stamps = transformer.extract_timestamps(self._df, self.config.time_variable)

            # Scale and normalize data
            if self.config.time_series_scaler:
                scaled_data = (
                    scaler.fit_transform(load_data)
                    if self.config.is_training_set
                    else scaler.transform(load_data)
                )
                scaled_data = (scaled_data - np.mean(scaled_data)) / np.std(scaled_data)
            else:
                scaled_data = load_data

            # Generate features for each timestamp
            sequence_rows = []
            for load_value, time_stamp in zip(scaled_data, time_stamps):
                features = [load_value]  # Main value first
                
                if self.config.include_time_information:
                    timestamp_pd = pd.Timestamp(time_stamp)
                    # Add cyclical time encodings
                    features.extend([
                        np.sin(2 * np.pi * timestamp_pd.hour / 24),
                        np.cos(2 * np.pi * timestamp_pd.hour / 24),
                        np.sin(2 * np.pi * timestamp_pd.dayofweek / 7),
                        np.cos(2 * np.pi * timestamp_pd.dayofweek / 7),
                        np.sin(2 * np.pi * timestamp_pd.month / 12),
                        np.cos(2 * np.pi * timestamp_pd.month / 12)
                    ])

                sequence_rows.append(features)

            # Convert to tensor with gradient tracking enabled
            self.rows = torch.tensor(
                np.array(sequence_rows, dtype=np.float32),
                requires_grad=True,
                dtype=torch.float32
            )

            # Print debug information
            self.logger.debug("Feature generation complete", {
                "total_sequences": len(self.rows),
                "features_per_sequence": self.rows.shape[-1],
                "value_range": f"[{self.rows.min():.2f}, {self.rows.max():.2f}]",
                "requires_grad": self.rows.requires_grad
            })

            # Validate tensor
            if torch.isnan(self.rows).any() or torch.isinf(self.rows).any():
                raise ValueError("Invalid values detected in final tensor")

        except Exception as e:
            self.logger.error("Error preparing time series data", {
                "error": str(e),
                "dataframe_info": str(self._df.info())
            })
            raise