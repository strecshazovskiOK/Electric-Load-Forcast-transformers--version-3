# models/registry/model_registry.py
from typing import Dict, Type, Any
import logging
from models.base.base_model import BaseModel
from models.registry.model_types import ModelType
from models.registry.base_registry import BaseModelRegistry

class ModelRegistry(BaseModelRegistry):
    """
    Central registry for model types implementing the Singleton pattern.
    This class maintains a single source of truth for model registration.
    """
    # Singleton instance
    _instance = None
    
    # Registry storage
    _registry: Dict[ModelType, Type[BaseModel]] = {}
    
    def __new__(cls):
        """Ensure only one instance exists using Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize instance attributes."""
        if not hasattr(self, '_logger'):
            # Initialize logging
            logging.basicConfig(level=logging.INFO)
            self._logger = logging.getLogger(__name__)

    def register_model(self, model_type: ModelType, model_class: Type[BaseModel]) -> None:
        """
        Register a model class for a specific model type with validation.
        
        Args:
            model_type: Type identifier for the model
            model_class: The model class to register
            
        Raises:
            TypeError: If model_class doesn't inherit from BaseModel
            ValueError: If model_type is already registered
        """
        # Validate model class inheritance
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"Model class {model_class.__name__} must inherit from BaseModel")
            
        # Check for duplicate registration
        if model_type in self._registry:
            self._logger.warning(f"Model type {model_type} is being re-registered")
            
        # Register the model
        self._registry[model_type] = model_class
        self._logger.info(f"Registered model {model_class.__name__} for type {model_type}")

    def get_model_class(self, model_type: ModelType) -> Type[BaseModel]:
        """
        Get the registered model class for a model type.
        
        Args:
            model_type: Type of model to retrieve
            
        Returns:
            Registered model class
            
        Raises:
            ValueError: If model type is not registered
        """
        if model_type not in self._registry:
            available_types = list(self._registry.keys())
            raise ValueError(
                f"No model registered for type: {model_type}. "
                f"Available types: {available_types}"
            )
        return self._registry[model_type]

    def create_model(self, model_type: ModelType, config: Dict[str, Any]) -> BaseModel:
        """
        Create and initialize a model instance.
        
        Args:
            model_type: Type of model to create
            config: Configuration for model initialization
            
        Returns:
            Initialized model instance
            
        Raises:
            ValueError: If model creation fails
        """
        try:
            model_class = self.get_model_class(model_type)
            model = model_class(config)
            self._logger.info(f"Created model instance of type {model_type}")
            return model
        except Exception as e:
            raise ValueError(f"Failed to create model of type {model_type}: {str(e)}") from e

    def get_registered_models(self) -> Dict[ModelType, Type[BaseModel]]:
        """Get a copy of the registry mapping."""
        return self._registry.copy()
