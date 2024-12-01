# training/trainers/__init__.py
from .neural_net_trainer import NeuralNetTrainer
from .transformer_trainer import TransformerTrainer

__all__ = ['NeuralNetTrainer', 'TransformerTrainer']