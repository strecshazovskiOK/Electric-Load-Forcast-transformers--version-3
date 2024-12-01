# experiments/experiment.py
import argparse
import datetime
import json
import os
from typing import Optional

from models.base.base_wrapper import BaseWrapper
from training.reports.training_report import TrainingReport  # Updated import path
from evaluation.results import EvaluationResult

class Experiment:
    """
    Collects all important information needed for reproducing and analysing results of trained model.
    """

    def __init__(
            self,
            model_wrapper: BaseWrapper,
            evaluation: EvaluationResult,  # Changed from Evaluation to EvaluationResult
            training_config: argparse.Namespace,
            training_report: Optional[TrainingReport],  # Use renamed class
            training_time: float,
            test_time: float
    ):
        self.model_wrapper = model_wrapper
        self.evaluation = evaluation
        self.training_config = training_config
        self.training_report = training_report
        self.training_time = training_time
        self.test_time = test_time

        # Define constants as class attributes
        self.JSON_FILE_ENDING = '.json'
        self.EXPERIMENTS_DIRECTORY = 'experiments'
        self.FINAL_EXPERIMENTS_DIRECTORY = ''

    @classmethod
    def from_json(cls, data: dict):
        """
        Creates an Experiment instance from JSON data.
        """
        # Convert training config back to Namespace
        training_config = argparse.Namespace(**data['trainingConfig'])
        
        # Reconstruct model wrapper (you might need to implement a from_str method in BaseWrapper)
        model_wrapper = BaseWrapper.from_str(data['modelWrapper'])
        
        # Create EvaluationResult from serialized data
        evaluation = EvaluationResult.from_dict(data['evaluation'])
        
        # Create TrainingReport from serialized data
        training_report = TrainingReport.from_dict(data['trainingReport']) if data['trainingReport'] else None
        
        return cls(
            model_wrapper=model_wrapper,
            evaluation=evaluation,
            training_config=training_config,
            training_report=training_report,
            training_time=data['training_time'],
            test_time=data['test_time']
        )

    def save_to_json_file(self) -> None:
        """
        Saves the experiment data to a json file. The name of the file is specified by the executed model and the time
        of execution. The data is serialized before storing.
        """
        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        experiment_name = f"{str(self.model_wrapper.model_type)}_{date}"
        print(experiment_name)

        result = {
            'experimentName': experiment_name,
            'modelType': str(self.model_wrapper.model_type),
            'modelWrapper': str(self.model_wrapper),
            'trainingConfig': self.training_config.__dict__,
            'trainingReport': self.training_report.serialize() if self.training_report else None,
            'evaluation': self.evaluation.serialize(),
            'training_time': self.training_time,
            'test_time': self.test_time
        }

        file_path = os.path.join(
            self.EXPERIMENTS_DIRECTORY,
            self.FINAL_EXPERIMENTS_DIRECTORY,
            experiment_name + self.JSON_FILE_ENDING
        )

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as fp:
            json.dump(result, fp, indent=4)

    def __str__(self):
        return (
            f'Model type: {str(self.model_wrapper.model_type)}\n'
            f'Model architecture: {str(self.model_wrapper)}\n'
            f'Training configuration: {str(self.training_config)}\n'
            f'Training report: {str(self.training_report)}\n'
            f'Evaluation: {str(self.evaluation)}'
        )