# models/registry/factory.py
from typing import Any, Dict, Optional, Type, ClassVar
from sklearn.base import BaseEstimator, RegressorMixin

from models.base.base_model import BaseModel
from models.wrappers.pytorch_wrapper import PyTorchWrapper
from models.registry.model_types import ModelType
from models.registry.model_registry import ModelRegistry
from models.architectures.transformers.resolution_specific import get_transformer_for_resolution
from ..interfaces import WrapperInterface

class ModelSklearnAdapter(BaseEstimator, RegressorMixin):
    """Adapter to make our models compatible with sklearn interface."""
    
    def __init__(self, model: BaseModel, model_type: ModelType, config: Dict[str, Any]):
        self.model = model
        self.model_type = model_type
        self.config = config

    def fit(self, X, y=None):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def partial_fit(self, X, y=None):
        """Incrementally fit the model if supported."""
        if hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X, y)
        else:
            # Fallback to regular fit if partial_fit is not implemented
            self.fit(X, y)
        return self

class ModelFactory:
    """
    Factory for creating and managing model instances. This class provides centralized
    model creation and registration functionality using a decorator pattern.
    """
    # Class-level registry instance
    _registry: ClassVar[ModelRegistry] = ModelRegistry()
    
    @classmethod
    def register(cls, model_type: ModelType):
        """
        Class method decorator for registering model implementations.
        
        Usage:
            @ModelFactory.register(ModelType.LINEAR_REGRESSION)
            class LinearRegression(BaseModel):
                ...
        
        Args:
            model_type: The type of model to register
            
        Returns:
            A decorator function that registers the model class
        """
        def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
            # Register the model class with validation
            if not issubclass(model_class, BaseModel):
                raise TypeError(f"Model class {model_class.__name__} must inherit from BaseModel")
            cls._registry.register_model(model_type, model_class)
            return model_class
        return decorator

    @classmethod
    def create_base_model(cls, model_type: ModelType, config: Dict[str, Any]) -> BaseModel:
        """
        Create a raw model instance based on type and configuration.
        
        Args:
            model_type: Type of model to create
            config: Configuration dictionary for model initialization
            
        Returns:
            Instantiated BaseModel
            
        Raises:
            ValueError: If model type is unknown or configuration is invalid
        """
        try:
            # Handle resolution-specific transformer types
            if model_type in {
                ModelType.SUBHOURLY_TRANSFORMER,
                ModelType.HOURLY_TRANSFORMER,
                ModelType.DAILY_TRANSFORMER,
                ModelType.MONTHLY_TRANSFORMER
            }:
                # Get appropriate transformer class based on resolution
                transformer_class = get_transformer_for_resolution(config['forecast_resolution_minutes'])
                return transformer_class(config)
            
            # Create standard model from registry
            return cls._registry.create_model(model_type, config)
            
        except KeyError as e:
            registered_models = cls.get_registered_models()
            raise ValueError(
                f"Failed to create model of type {model_type}. "
                f"Available types: {list(registered_models.keys())}"
            ) from e
        except Exception as e:
            raise ValueError(f"Error creating model: {str(e)}") from e

    @classmethod
    def create(cls, model_type: ModelType, config: Dict[str, Any], 
               wrapper_type: Optional[str] = None) -> WrapperInterface:
        """
        Create a wrapped model instance ready for training and inference.
        
        Args:
            model_type: Type of model to create
            config: Model configuration dictionary
            wrapper_type: Optional wrapper type ('pytorch' or 'sklearn')
            
        Returns:
            Wrapped model instance
            
        Raises:
            ValueError: If configuration or wrapper type is invalid
        """
        try:
            # Create base model
            base_model = cls.create_base_model(model_type, config)
            
            # Choose appropriate wrapper
            if wrapper_type != 'sklearn':
                return PyTorchWrapper(
                    model=base_model,
                    model_type=model_type,
                    config=config
                )
                
            # Create sklearn wrapper if requested
            from models.wrappers.sklearn_wrapper import SklearnWrapper
            adapter = ModelSklearnAdapter(base_model, model_type, config)
            return SklearnWrapper(
                model=adapter,
                model_type=model_type,
                config=config
            )
            
        except Exception as e:
            raise ValueError(f"Error creating wrapped model: {str(e)}") from e

    @classmethod
    def get_registered_models(cls) -> Dict[ModelType, Type[BaseModel]]:
        """Get dictionary of all registered model types and their classes."""
        return cls._registry.get_registered_models()

    @classmethod
    def get_resolution_models(cls) -> Dict[str, ModelType]:
        """Get mapping of resolution names to their model types."""
        return {
            'subhourly': ModelType.SUBHOURLY_TRANSFORMER,
            'hourly': ModelType.HOURLY_TRANSFORMER,
            'daily': ModelType.DAILY_TRANSFORMER,
            'monthly': ModelType.MONTHLY_TRANSFORMER
        }