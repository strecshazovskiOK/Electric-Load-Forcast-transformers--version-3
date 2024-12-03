import numpy as np
import pandas as pd
import torch
from typing import Tuple, List, Any

from data_loading.base.base_dataset import BaseDataset, DatasetConfig
from data_loading.features.time_features import CyclicalTimeFeature, WorkdayFeature
from data_loading.preprocessing.data_scaler import DataScaler
from data_loading.preprocessing.data_transformer import DataTransformer

class TransformerDataset(BaseDataset):
    """Dataset implementation for transformer models with proper sequence handling"""

    def __init__(self, df: pd.DataFrame, config: DatasetConfig):
        super().__init__(df, config)
        self.rows: torch.Tensor = torch.empty(0)
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

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Get a data item formatted for transformer training.
        
        Args:
            index: Position in dataset to retrieve
            
        Returns:
            ((src_seq, tgt_input_seq), target_seq):
            - src_seq: Source sequence for encoder [seq_len, features]
            - tgt_input_seq: Target input sequence for decoder [seq_len-1, features]
            - target_seq: Target values to predict [seq_len-1, 1]
        """
        if self.rows is None:
            raise ValueError("Dataset not properly initialized")

        # Calculate window sizes
        input_window = self.config.window_size
        forecast_window = self.config.horizon_size

        try:
            # Get source sequence for encoder
            src_seq = self.rows[index:index + input_window]
            
            # Get target sequence
            full_target = self.rows[index + input_window:index + input_window + forecast_window]
            
            # Prepare sequences for teacher forcing:
            # - Remove last timestep from target input (shifted right)
            # - Remove first timestep from target output (shifted left)
            tgt_input_seq = full_target[:-1]
            target_seq = full_target[1:, 0:1]  # Only keep energy consumption value

            # Debug information
            if self._debug_counter % self._debug_frequency == 0:
                print(f"\nDebug - Dataset __getitem__ (sample {self._debug_counter}):")
                print(f"Source sequence shape: {src_seq.shape}")
                print(f"Target input sequence shape: {tgt_input_seq.shape}")
                print(f"Target sequence shape: {target_seq.shape}")
                print(f"Source sequence range: [{src_seq.min():.2f}, {src_seq.max():.2f}]")
                print(f"Target sequence range: [{target_seq.min():.2f}, {target_seq.max():.2f}]")
            
            self._debug_counter += 1

            # Add batch dimension if needed
            if len(src_seq.shape) == 2:
                src_seq = src_seq.unsqueeze(0)
                tgt_input_seq = tgt_input_seq.unsqueeze(0)
                target_seq = target_seq.unsqueeze(0)

            return (src_seq, tgt_input_seq), target_seq
            
        except Exception as e:
            print(f"Error in __getitem__ at index {index}: {str(e)}")
            print(f"Dataset size: {len(self.rows)}")
            print(f"Requested windows: input={input_window}, forecast={forecast_window}")
            raise

    def _prepare_time_series_data(self) -> None:
        """Prepare time series data for transformer model input"""
        print(f"\nDebug - TransformerDataset preparation:")
        print(f"Initial DataFrame shape: {self._df.shape}")
        
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

            # Convert to tensor and store
            self.rows = torch.tensor(np.array(sequence_rows, dtype=np.float32))
            
            # Print debug information
            print(f"Debug - Feature generation:")
            print(f"  Total sequences: {len(self.rows)}")
            print(f"  Features per sequence: {self.rows.shape[-1]}")
            print(f"  Value range: [{self.rows.min():.2f}, {self.rows.max():.2f}]")

            # Validate tensor
            if torch.isnan(self.rows).any() or torch.isinf(self.rows).any():
                raise ValueError("Invalid values detected in final tensor")

        except Exception as e:
            print(f"\nError preparing time series data: {str(e)}")
            print(f"DataFrame info:")
            print(self._df.info())
            raise