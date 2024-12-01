# data_loading/factories/dataset_factory.py
from typing import Type
import pandas as pd

from data_loading.base.base_dataset import BaseDataset, DatasetConfig
from data_loading.datasets.standard_dataset import StandardDataset
from data_loading.datasets.transformer_dataset import TransformerDataset




class DatasetFactory:
    """Factory for creating dataset instances"""

    _dataset_map: 'dict[str, Type[BaseDataset]]' = {
        'standard': StandardDataset,
        'transformer': TransformerDataset
    }

    @classmethod
    def create_dataset(cls, dataset_type: str, df: pd.DataFrame, config: DatasetConfig) -> BaseDataset:
        """Create a dataset instance of the specified type"""
        dataset_class = cls._dataset_map.get(dataset_type.lower())
        if dataset_class is None:
            raise ValueError(
                f"Unknown dataset type: {dataset_type}. "
                f"Available types: {list(cls._dataset_map.keys())}"
            )
        return dataset_class(df, config)

    @classmethod
    def register_dataset(
            cls,
            name: str,
            dataset_class: Type[BaseDataset]
    ) -> None:
        """Register a new dataset type"""
        cls._dataset_map[name.lower()] = dataset_class