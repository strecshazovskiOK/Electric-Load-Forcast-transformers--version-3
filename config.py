# # config.py
# from dataclasses import dataclass
# from datetime import date
# from pathlib import Path
# from typing import Union
# from sklearn.preprocessing import StandardScaler
# from data_loading.base.base_dataset import DatasetConfig
# from models.config.model_config import ModelConfig
# from training.config import TransformerTrainingConfig
# from pipeline.config.pipeline_config import PipelineConfig
# from models.registry.model_types import ModelType
# def create_transformer_config(data_path: Union[str, Path]) -> PipelineConfig:
#     """Create configuration for transformer model training."""
    
#     dataset_config = DatasetConfig(
#         time_variable='utc_timestamp',
#         target_variable='DE_KN_residential1_grid_import',
#         time_series_window_in_hours=24,
#         forecasting_horizon_in_hours=12,
#         is_single_time_point_prediction=False,
#         include_time_information=True,
#         time_series_scaler=StandardScaler(),
#         is_training_set=True,
#         labels_count=12,
#         one_hot_time_variables=False
#     )

#     model_config = ModelConfig(
#         model_type=ModelType.VANILLA_TRANSFORMER,
#         input_features=1,
#         d_model=64,
#         n_heads=4,
#         n_encoder_layers=3,
#         n_decoder_layers=3,
#         d_ff=256,
#         dropout=0.1
#     )

#     training_config = TransformerTrainingConfig(
#         learning_rate=0.001,
#         max_epochs=100,
#         use_early_stopping=True,
#         early_stopping_patience=10,
#         batch_size=32,
#         device='cuda',
#         transformer_labels_count=12,
#         forecasting_horizon=12,
#         transformer_use_teacher_forcing=True
#     )

#     return PipelineConfig(
#         dataset_config=dataset_config,
#         model_config=model_config,
#         training_config=training_config,
#         data_path=Path(data_path),
#         model_save_path=Path('models/transformer'),
#         experiment_save_path=Path('experiments'),
#         train_dates=(date(2015, 10, 29), date(2016, 10, 28)),
#         val_dates=(date(2016, 10, 29), date(2017, 1, 28)),
#         test_dates=(date(2017, 1, 29), date(2017, 3, 12))
#     )