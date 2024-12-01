# models/registry/factory.py
from typing import Any, Dict, Type, Callable

from models.base.base_model import BaseModel
from models.wrappers.pytorch_wrapper import PyTorchWrapper

from .model_types import ModelType
from ..interfaces import WrapperInterface

class ModelFactory:
    """Factory for creating model instances."""
    _registry: Dict[ModelType, Type[BaseModel]] = {}

    @classmethod
    def register(cls, model_type: ModelType) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
        """Register a model in the factory."""
        def decorator(model_cls: Type[BaseModel]) -> Type[BaseModel]:
            print(f"Registering model {model_cls.__name__} for type {model_type}")
            cls._registry[model_type] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create_base_model(cls, model_type: ModelType, config: Dict[str, Any]) -> BaseModel:
        """Create a raw model instance."""
        if not cls._registry:
            print("Warning: Model registry is empty. Make sure models are imported and registered.")
            raise ValueError("No models registered in factory")
        
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(cls._registry.keys())}")
        
        model_class = cls._registry[model_type]
        return model_class(config)

    @classmethod
    def create(cls, model_type: ModelType, config: Dict[str, Any]) -> WrapperInterface:
        """Create a wrapped model instance."""
        base_model = cls.create_base_model(model_type, config)

        # Import here to avoid circular imports
        return PyTorchWrapper(
            model=base_model,
            model_type=model_type,
            config=config
        )

    @classmethod
    def get_registered_models(cls) -> Dict[ModelType, Type[BaseModel]]:
        """Get all registered model types."""
        return cls._registry.copy()