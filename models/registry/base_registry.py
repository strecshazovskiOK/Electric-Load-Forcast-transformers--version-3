
from typing import Dict, Type, Any
from models.base.base_model import BaseModel
from models.registry.model_types import ModelType

class BaseModelRegistry:
    """Base interface for model registration."""
    def register_model(self, model_type: ModelType, model_class: Type[BaseModel]) -> None:
        raise NotImplementedError
        
    def get_model_class(self, model_type: ModelType) -> Type[BaseModel]:
        raise NotImplementedError

    def create_model(self, model_type: ModelType, config: Dict[str, Any]) -> BaseModel:
        raise NotImplementedError