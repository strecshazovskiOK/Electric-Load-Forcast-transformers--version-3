# data_loading/datasets/transformer_dataset.py
import numpy as np
import pandas as pd
import torch
from typing import Tuple, List, Any

from data_loading.base.base_dataset import BaseDataset, DatasetConfig
from data_loading.features.time_features import CyclicalTimeFeature, WorkdayFeature
from data_loading.preprocessing.data_scaler import DataScaler
from data_loading.preprocessing.data_transformer import DataTransformer

class TransformerDataset(BaseDataset):
    """Dataset implementation for transformer models"""

    def __init__(self, df: pd.DataFrame, config: DatasetConfig):
        super().__init__(df, config)
        self.rows: torch.Tensor = torch.empty(0)
        self._prepare_time_series_data()
        self._debug_counter = 0  # Add debug counter
        self._debug_frequency = 10000  # Print debug info every 1000 samples

    def __len__(self) -> int:
        if self.rows is None:
            return 0
        total_window = (self.config.time_series_window_in_hours + 
                       self.config.forecasting_horizon_in_hours) * self.config.points_per_hour
        return max(0, len(self.rows) - total_window)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a data item by index"""
        if self.rows is None:
            raise ValueError("Dataset not properly initialized")

        # Adjust window sizes to account for 15-minute intervals
        input_window = self.config.time_series_window_in_hours * self.config.points_per_hour
        forecast_window = self.config.forecasting_horizon_in_hours * self.config.points_per_hour

        input_seq = self.rows[
                    index:index + input_window
                    ]
        
        target_seq = self.rows[
                    index + input_window:
                    index + input_window + forecast_window
                    ]

        # Debug info
        if self._debug_counter % self._debug_frequency == 0:
            print(f"\nDebug - Dataset __getitem__ (sample {self._debug_counter}):")
            print(f"Input sequence shape: {input_seq.shape}")
            print(f"Target sequence shape: {target_seq.shape}")
            print(f"Input sequence value range: [{input_seq.min():.2f}, {input_seq.max():.2f}]")
            print(f"Target sequence value range: [{target_seq.min():.2f}, {target_seq.max():.2f}]")
        
        self._debug_counter += 1
        return input_seq, target_seq

    def _prepare_time_series_data(self) -> None:
        """Prepare time series data for model input"""
        print(f"\nDebug - TransformerDataset preparation:")
        print(f"Initial DataFrame shape: {self._df.shape}")
        
        if len(self._df) == 0:
            raise ValueError("Empty dataframe provided")

        # Initialize processors
        scaler = DataScaler(self.config.time_series_scaler)
        transformer = DataTransformer()  # Add this line to instantiate DataTransformer

        # Make sure we have enough data points
        min_required = (self.config.time_series_window_in_hours + 
                    self.config.forecasting_horizon_in_hours) * 4  # 4 points per hour
        print(f"Debug - Required data points: {min_required}")
        print(f"Debug - Available data points: {len(self._df)}")
        
        if len(self._df) < min_required:
            raise ValueError(f"Not enough data points. Need at least {min_required}, got {len(self._df)}")

        try:
            # Extract raw data (already in 15-min intervals)
            load_data = np.array(self._df[self.config.target_variable])
            print(f"Debug - Load data shape: {load_data.shape}")
            
            # Scale data
            if self.config.time_series_scaler:
                scaled_data = (
                    scaler.fit_transform(load_data)
                    if self.config.is_training_set
                    else scaler.transform(load_data)
                )
                # Add normalization
                scaled_data = (scaled_data - np.mean(scaled_data)) / np.std(scaled_data)
                print(f"Debug - After normalization range: [{scaled_data.min():.2f}, {scaled_data.max():.2f}]")
            else:
                scaled_data = load_data

            # Add validation check
            if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
                raise ValueError("Invalid values detected after scaling")

            # Extract timestamps and prepare features using the instantiated transformer
            time_stamps = transformer.extract_timestamps(
                self._df,
                self.config.time_variable
            )
            print(f"Debug - Time stamps shape: {time_stamps.shape}")

            # Initialize feature generators
            hour_feature = CyclicalTimeFeature(24)
            week_feature = CyclicalTimeFeature(53)
            workday_feature = WorkdayFeature()

            sequence_rows = []

            # Generate features for each timestamp
            print(f"Debug - Generating features for {len(scaled_data)} time points")
            for idx, (load_value, time_stamp) in enumerate(zip(scaled_data, time_stamps)):
                features = [load_value]

                if self.config.include_time_information:
                    timestamp_pd = pd.Timestamp(time_stamp)
                    features.extend(hour_feature.generate(pd.Series([timestamp_pd.hour])))
                    features.extend(week_feature.generate(pd.Series([timestamp_pd.isocalendar()[1]])))
                    features.extend(workday_feature.generate(timestamp_pd))

                sequence_rows.append(features)

            self.rows = torch.tensor(np.array(sequence_rows, dtype=np.float32))
            print(f"Debug - Final rows tensor shape: {self.rows.shape}")

            # Add final validation
            if torch.isnan(self.rows).any() or torch.isinf(self.rows).any():
                raise ValueError("Invalid values detected in final tensor")
            
            print(f"Debug - Final value range: [{self.rows.min():.2f}, {self.rows.max():.2f}]")

            # Set input and target tensors
            self.prepared_time_series_input = self.rows[
                :len(self.rows) - self.config.forecasting_horizon_in_hours
            ]
            self.prepared_time_series_target = self.rows[
                self.config.time_series_window_in_hours:
            ]

            print(f"Debug - Final shapes:")
            print(f"  Input tensor: {self.prepared_time_series_input.shape if self.prepared_time_series_input is not None else None}")
            print(f"  Target tensor: {self.prepared_time_series_target.shape if self.prepared_time_series_target is not None else None}")

        except Exception as e:
            print(f"\nError preparing time series data: {str(e)}")
            print(f"DataFrame info:")
            print(self._df.info())
            raise