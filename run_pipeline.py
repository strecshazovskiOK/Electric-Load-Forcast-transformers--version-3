# # run_pipeline.py
# from pathlib import Path
# from sklearn.preprocessing import StandardScaler

# from data_loading.base.base_dataset import DatasetConfig
# from models.registry.model_types import ModelType
# from pipeline.implementations.time_series_pipeline import TimeSeriesPipeline
# from pipeline.utils.config_utils import create_pipeline_config
# from training.config import TrainingConfig


# def _run_pipeline_and_evaluate(pipeline):
#     experiment = pipeline.run()
#     print("Pipeline executed successfully!")
#     print(f"Training time: {experiment.training_time:.2f} seconds")
#     print(f"Test time: {experiment.test_time:.2f} seconds")
#     print("\nModel Evaluation Metrics:")
#     print(experiment.evaluation.total_metrics)

# def main():
#     # 1. Define Dataset Configuration
#     dataset_config = DatasetConfig(
#         time_variable="utc_timestamp",
#         target_variable="DE_KN_residential1_grid_import",
#         time_series_window_in_hours=24,
#         forecasting_horizon_in_hours=24,
#         is_single_time_point_prediction=False,
#         include_time_information=True,
#         time_series_scaler=StandardScaler(),
#         is_training_set=True,
#         labels_count=1,
#         one_hot_time_variables=False
#     )

#     # 2. Define Model Parameters
#     model_params = {
#         "input_features": 10,  # Adjust based on your feature count
#         "output_dim": 1,
#         "hidden_dims": [64, 32],
#         "d_model": 512,
#         "n_heads": 8,
#         "n_encoder_layers": 3,
#         "n_decoder_layers": 3,
#         "d_ff": 2048,
#         "dropout": 0.1
#     }

#     # 3. Define Training Parameters
#     training_params = {
#         "learning_rate": 0.001,
#         "max_epochs": 100,
#         "batch_size": 32,
#         "use_early_stopping": True,
#         "early_stopping_patience": 10,
#         # "transformer_labels_count": 1,
#         # "transformer_use_teacher_forcing": False,
#         # "transformer_use_auto_regression": False,
#         # "learning_rate_scheduler_step": 30,
#         # "learning_rate_scheduler_gamma": 0.1,
#         # "attention_dropout": 0.1
#     }

#     # 4. Create Pipeline Configuration
#     config = create_pipeline_config(
#         data_path=Path("data/household_15min.csv"),
#         model_type=ModelType.TIME_SERIES_TRANSFORMER,
#         model_params=model_params,
#         training_params=training_params,
#         dataset_params=dataset_config.__dict__
#     )

#     # 5. Create and Run Pipeline
#     pipeline = TimeSeriesPipeline(config)
#     try:
#         _run_pipeline_and_evaluate(pipeline)
#     except Exception as e:
#         print(f"Pipeline execution failed: {str(e)}")



# if __name__ == "__main__":
#     main()