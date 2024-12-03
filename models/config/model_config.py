# models/config/model_config.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from models.registry.model_types import ModelType
from data_loading.types.interval_types import TimeInterval, IntervalConfig

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    # Core model identification
    model_type: ModelType
    
    # Time resolution and sequence configuration
    input_resolution_minutes: int
    forecast_resolution_minutes: int
    lookback_periods: int
    forecast_periods: int
    
    
    # Model architecture parameters
    input_features: int
    
    output_features: int = 1
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    
    max_sequence_length: int = 5000
    
    # Time series specific parameters
    value_features: int = 1  # Main time series value
    time_features: Optional[int] = None  # Additional time-based features
    kernel_size: int = 3  # For convolutional attention
    batch_first: bool = True  # Handle batch dimension first
    
    # Neural network specific
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    activation: str = 'relu'

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_epochs: int = 100
    optimizer: str = 'adam'
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    scheduler: Optional[str] = None
    scheduler_config: Dict[str, Any] = field(default_factory=dict)
    criterion: str = 'mse'
    criterion_config: Dict[str, Any] = field(default_factory=dict)

    # Device
    device: str = 'cuda'

    # Cache for computed values
    _interval_config: Optional[IntervalConfig] = field(init=False, default=None)
    _input_points_per_period: Optional[int] = field(init=False, default=None)
    _forecast_points_per_period: Optional[int] = field(init=False, default=None)

    def __post_init__(self):
        """Post-initialization processing with enhanced feature validation."""
        self._validate_resolutions()
        self._validate_features()  # Add this line
        self._initialize_interval_config()
        self._compute_points_per_period()

        
        # If time_features not specified, calculate from input features
        if self.time_features is None:
            self.time_features = max(0, self.input_features - self.value_features)
            
    def _validate_features(self) -> None:
            """Validate feature dimensions."""
            if self.time_features is None:
                self.time_features = self.input_features - self.value_features
                
            total_features = self.value_features + self.time_features
            if total_features != self.input_features:
                raise ValueError(
                    f"Feature dimension mismatch! Total features ({self.input_features}) must equal "
                    f"value_features ({self.value_features}) + time_features ({self.time_features})"
                )

    def _validate_resolutions(self) -> None:
        """Validate input and forecast resolutions."""
        # Validate input resolution
        if 60 % self.input_resolution_minutes != 0 and self.input_resolution_minutes % 60 != 0:
            raise ValueError(f"Invalid input resolution: {self.input_resolution_minutes} minutes")
            
        # Validate forecast resolution
        if 60 % self.forecast_resolution_minutes != 0 and self.forecast_resolution_minutes % 60 != 0:
            raise ValueError(f"Invalid forecast resolution: {self.forecast_resolution_minutes} minutes")
            
        # Validate that forecast resolution is not finer than input resolution
        if self.forecast_resolution_minutes < self.input_resolution_minutes:
            raise ValueError(
                f"Forecast resolution ({self.forecast_resolution_minutes} min) cannot be finer than "
                f"input resolution ({self.input_resolution_minutes} min)"
            )

    def _initialize_interval_config(self) -> None:
        """Initialize the interval configuration."""
        self._interval_config = IntervalConfig(
            interval_type=self._determine_interval_type(),
            lookback_periods=self.lookback_periods,
            forecast_periods=self.forecast_periods
        )

    def _compute_points_per_period(self) -> None:
        """Compute points per period for both input and forecast resolutions."""
        self._input_points_per_period = (
            60 // self.input_resolution_minutes 
            if self.input_resolution_minutes <= 60 
            else 1
        )
        self._forecast_points_per_period = (
            60 // self.forecast_resolution_minutes 
            if self.forecast_resolution_minutes <= 60 
            else 1
        )

    def _determine_interval_type(self) -> TimeInterval:
        """Determine the interval type based on forecast resolution."""
        resolution = self.forecast_resolution_minutes
        if resolution <= 15:
            return TimeInterval.FIFTEEN_MIN
        elif resolution <= 60:
            return TimeInterval.HOURLY
        elif resolution <= 1440:  # 24 hours
            return TimeInterval.DAILY
        else:
            return TimeInterval.MONTHLY

    @property
    def sequence_length(self) -> int:
        """Get total sequence length for model input."""
        if self._input_points_per_period is None:
            self._compute_points_per_period()
        if self._input_points_per_period is None:  # double-check after computation
            raise ValueError("Input points per period not properly initialized")
        return self.lookback_periods * self._input_points_per_period

    @property
    def forecast_length(self) -> int:
        """Get total forecast length for model output."""
        if self._forecast_points_per_period is None:
            self._compute_points_per_period()
        if self._forecast_points_per_period is None:  # double-check after computation
            raise ValueError("Forecast points per period not properly initialized")
        return self.forecast_periods * self._forecast_points_per_period

    def needs_resampling(self) -> bool:
        """Check if input data needs resampling for forecasting."""
        return self.input_resolution_minutes != self.forecast_resolution_minutes

    def get_resampling_factor(self) -> float:
        """Get the factor by which to resample the data."""
        return self.forecast_resolution_minutes / self.input_resolution_minutes

    def get_interval_config(self) -> Optional[IntervalConfig]:
        """Get the interval configuration."""
        return self._interval_config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        base_dict = {
            'model_type': self.model_type,
            'input_resolution_minutes': self.input_resolution_minutes,
            'forecast_resolution_minutes': self.forecast_resolution_minutes,
            'lookback_periods': self.lookback_periods,
            'forecast_periods': self.forecast_periods,
            'input_features': self.input_features,
            'output_features': self.output_features,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_encoder_layers': self.n_encoder_layers,
            'n_decoder_layers': self.n_decoder_layers,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
            'value_features': self.value_features,
            'time_features': self.time_features,
            'kernel_size': self.kernel_size,
            'batch_first': self.batch_first,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'max_epochs': self.max_epochs,
            'optimizer': self.optimizer,
            'criterion': self.criterion,
            'device': self.device
        }
        
        # Add configuration dictionaries if they're not empty
        if self.optimizer_config:
            base_dict['optimizer_config'] = self.optimizer_config
        if self.scheduler_config:
            base_dict['scheduler_config'] = self.scheduler_config
        if self.criterion_config:
            base_dict['criterion_config'] = self.criterion_config
            
        return base_dict

    @classmethod
    def get_default_config(
        cls,
        model_type: ModelType,  # Changed from str to ModelType
        input_resolution_minutes: int = 15,
        forecast_resolution_minutes: int = 15,
        input_features: int = 7
    ) -> 'ModelConfig':
        """Get default configuration for specified model type and resolutions.
        
        Args:
            model_type: Type of model to configure
            input_resolution_minutes: Resolution of input data in minutes
            forecast_resolution_minutes: Resolution of forecast in minutes
        
        Returns:
            ModelConfig with appropriate defaults for the specified model type and resolutions
        """
        # Base configuration common to all models
        base_config = {
            'model_type': model_type,
            'input_resolution_minutes': input_resolution_minutes,
            'forecast_resolution_minutes': forecast_resolution_minutes,
            'input_features': input_features,        # Ensure this is included
            'value_features': 1,                     # Single target value
            'time_features': input_features - 1,     # Remaining features are time features
        }


        # Resolution-specific defaults
        if forecast_resolution_minutes <= 60:  # Sub-hourly or hourly predictions
            base_config.update({
                'lookback_periods': 24,  # 24 periods
                'forecast_periods': 12,  # 12 periods ahead
            })
        elif forecast_resolution_minutes <= 1440:  # Daily predictions
            base_config.update({
                'lookback_periods': 7,   # 7 days of history
                'forecast_periods': 5,   # 5 days ahead
            })
        else:  # Monthly predictions
            base_config.update({
                'lookback_periods': 12,  # 12 months of history
                'forecast_periods': 3,   # 3 months ahead
            })

        # Model-specific configurations
        if model_type in {'vanilla_transformer', 'conv_transformer', 'informer', 'time_series_transformer'}:
            base_config.update({
                'd_model': 256,
                'n_heads': 8,
                'n_encoder_layers': 4,
                'n_decoder_layers': 4,
                'd_ff': 1024,
                'dropout': 0.2,
                'kernel_size': 3,
                'batch_first': True,
                
                # Training parameters optimized for time series
                'batch_size': 32,
                'learning_rate': 0.001,
                'max_epochs': 100,
                'optimizer': 'adamw',
                'optimizer_config': {
                    'weight_decay': 0.01,
                    'betas': (0.9, 0.98)
                },
                'scheduler': 'one_cycle',
                'scheduler_config': {
                    'pct_start': 0.3,
                    'div_factor': 25.0,
                    'final_div_factor': 1000.0
                }
            })

            # Resolution-specific model adjustments
            if forecast_resolution_minutes <= 15:  # Fine-grained predictions
                base_config.update({
                    'd_model': 128,  # Smaller model for more frequent data
                    'n_heads': 4,
                    'dropout': 0.3,  # More dropout for regularization
                    'batch_size': 64  # Larger batches for more frequent data
                })
            elif forecast_resolution_minutes >= 1440:  # Daily or longer
                base_config.update({
                    'd_model': 512,  # Larger model for long-term patterns
                    'n_heads': 16,
                    'dropout': 0.1,  # Less dropout for sparse data
                    'batch_size': 16  # Smaller batches for less frequent data
                })

        elif model_type == 'simple_neural_net':
            base_config.update({
                'hidden_dims': [64, 32],
                'activation': 'relu',
                'dropout': 0.1,
                'batch_size': 32,
                'learning_rate': 0.001
            })

        elif model_type == 'linear_regression':
            base_config.update({
                'optimizer': 'adam',
                'learning_rate': 0.01,
                'batch_size': 64
            })

        return cls(**base_config)