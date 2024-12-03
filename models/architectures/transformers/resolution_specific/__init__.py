# models/architectures/transformers/resolution_specific/__init__.py
from .base_resolution_transformer import BaseResolutionTransformer
from .subhourly_transformer import SubhourlyTransformer
from .hourly_transformer import HourlyTransformer
from .daily_transformer import DailyTransformer
from .monthly_transformer import MonthlyTransformer
from data_loading.types.interval_types import TimeInterval

def get_transformer_for_resolution(resolution_minutes: int) -> type:
    """
    Get appropriate transformer class for the given resolution.
    
    Args:
        resolution_minutes: The desired forecast resolution in minutes
        
    Returns:
        Appropriate transformer class for the resolution
    
    Raises:
        ValueError: If resolution is invalid or unsupported
    """
    if resolution_minutes <= 0:
        raise ValueError("Resolution must be positive")
    
    if resolution_minutes <= 60:
        return SubhourlyTransformer
    elif resolution_minutes <= 180:
        return HourlyTransformer
    elif resolution_minutes <= 2880:  # Up to 2 days
        return DailyTransformer
    else:
        return MonthlyTransformer

def create_resolution_transformer(
    resolution_minutes: int,
    config: dict
) -> BaseResolutionTransformer:
    """
    Create a transformer instance appropriate for the given resolution.
    
    Args:
        resolution_minutes: The desired forecast resolution in minutes
        config: Configuration dictionary for the transformer
        
    Returns:
        Configured transformer instance
    """
    transformer_class = get_transformer_for_resolution(resolution_minutes)
    return transformer_class(config)

__all__ = [
    'BaseResolutionTransformer',
    'SubhourlyTransformer',
    'HourlyTransformer',
    'DailyTransformer',
    'MonthlyTransformer',
    'get_transformer_for_resolution',
    'create_resolution_transformer'
]