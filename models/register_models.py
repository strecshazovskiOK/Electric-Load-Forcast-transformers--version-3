from models.registry.model_types import ModelType
from models.registry.factory import ModelFactory
from models.architectures.transformers.resolution_specific import (
    SubhourlyTransformer,
    HourlyTransformer,
    DailyTransformer,
    MonthlyTransformer
)

def register_models():
    """Register all model types with the registry."""
    registry = ModelFactory._registry
    registry.register_model(ModelType.SUBHOURLY_TRANSFORMER, SubhourlyTransformer)
    registry.register_model(ModelType.HOURLY_TRANSFORMER, HourlyTransformer)
    registry.register_model(ModelType.DAILY_TRANSFORMER, DailyTransformer)
    registry.register_model(ModelType.MONTHLY_TRANSFORMER, MonthlyTransformer)