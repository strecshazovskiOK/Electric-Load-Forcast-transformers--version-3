# data_loading/datasets/standard_dataset.py
import numpy as np
import pandas as pd
import torch
from typing import Tuple

from data_loading.base.base_dataset import BaseDataset, DatasetConfig
from data_loading.features.time_features import CyclicalTimeFeature, OneHotTimeFeature, WorkdayFeature
from data_loading.preprocessing.data_scaler import DataScaler
from data_loading.preprocessing.data_transformer import DataTransformer


class StandardDataset(BaseDataset):
    """Dataset implementation for standard ML models (linear regression, neural nets)"""

    def __init__(self, df, config: DatasetConfig):
        super().__init__(df, config)
        self.prepared_time_series_input: torch.Tensor = torch.zeros((0,))
        self.prepared_time_series_target: torch.Tensor = torch.zeros((0,))
        self._prepare_time_series_data()

    def __len__(self) -> int:
        return len(self.prepared_time_series_input)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.prepared_time_series_input is None or self.prepared_time_series_target is None:
            raise ValueError("Dataset not properly initialized")
        
        if index >= len(self.prepared_time_series_input):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self.prepared_time_series_input)}")
            
        return (
            self.prepared_time_series_input[index],
            self.prepared_time_series_target[index]
        )

    def _prepare_time_series_data(self) -> None:
        # Initialize data processor
        scaler = DataScaler(self.config.time_series_scaler)

        # Extract raw data (already in 15-min intervals)
        load_data = np.array(self._df[self.config.target_variable])
        time_stamps = np.array(self._df[self.config.time_variable])

        # Scale data
        scaled_load_data = (
            scaler.fit_transform(load_data)
            if self.config.is_training_set
            else scaler.transform(load_data)
        ) if self.config.time_series_scaler else load_data

        target_rows = []
        input_rows = []

        # Initialize feature generators
        hour_feature = (
            OneHotTimeFeature(24) if self.config.one_hot_time_variables
            else CyclicalTimeFeature(24)
        )
        month_feature = (
            OneHotTimeFeature(12) if self.config.one_hot_time_variables
            else CyclicalTimeFeature(53)
        )
        workday_feature = WorkdayFeature()

        # Generate input/target pairs
        for idx in range(
                self.config.time_series_window_in_hours,
                len(scaled_load_data) - self.config.forecasting_horizon_in_hours
        ):
            # Prepare target
            if self.config.is_single_time_point_prediction:
                target = [scaled_load_data[idx + self.config.forecasting_horizon_in_hours]]
            else:
                target = scaled_load_data[idx:idx + self.config.forecasting_horizon_in_hours]
            target_rows.append(target)

            # Prepare input features
            features = []

            # Add time series values
            features.extend(
                scaled_load_data[idx - self.config.time_series_window_in_hours:idx]
            )

            if self.config.include_time_information:
                prediction_datetime = pd.to_datetime(time_stamps[idx])

                # Add time features
                features.extend(hour_feature.generate(prediction_datetime.hour))
                features.extend(month_feature.generate(prediction_datetime.month - 1))

                # Add workday features
                if self.config.is_single_time_point_prediction:
                    features.extend(workday_feature.generate(prediction_datetime))
                else:
                    for t in time_stamps[idx:idx + self.config.forecasting_horizon_in_hours]:
                        features.extend(workday_feature.generate(t))

            input_rows.append(features)

        # Convert to tensors
        self.prepared_time_series_input = torch.tensor(
            np.array(input_rows),
            dtype=torch.float32
        )
        self.prepared_time_series_target = torch.tensor(
            np.array(target_rows),
            dtype=torch.float32
        )
        self.time_labels = time_stamps[
                        self.config.time_series_window_in_hours:
                        len(scaled_load_data) - self.config.forecasting_horizon_in_hours
                        ]