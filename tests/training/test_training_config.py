# tests/unit/test_training_config.py
from training.config import TrainingConfig, TransformerTrainingConfig, NeuralNetTrainingConfig

def test_base_training_config():
    """Test base training configuration initialization and attributes."""
    config = TrainingConfig(
        learning_rate=0.001,
        max_epochs=100,
        use_early_stopping=True,
        early_stopping_patience=10,
        batch_size=32,
        device='cuda'
    )

    assert config.learning_rate == 0.001
    assert config.max_epochs == 100
    assert config.use_early_stopping
    assert config.early_stopping_patience == 10
    assert config.batch_size == 32
    assert config.device == 'cuda'

def test_neural_net_training_config():
    """Test neural network specific training configuration."""
    config = NeuralNetTrainingConfig(
        learning_rate=0.001,
        max_epochs=100,
        use_early_stopping=True,
        early_stopping_patience=10,
        learning_rate_scheduler_step=30,
        learning_rate_scheduler_gamma=0.1,
        gradient_clipping=1.0,
        batch_size=32
    )

    # Test base attributes
    assert config.learning_rate == 0.001
    assert config.max_epochs == 100
    assert config.use_early_stopping
    assert config.early_stopping_patience == 10

    # Test neural net specific attributes
    assert config.learning_rate_scheduler_step == 30
    assert config.learning_rate_scheduler_gamma == 0.1
    assert config.gradient_clipping == 1.0
    assert config.batch_size == 32

def test_transformer_training_config():
    """Test transformer specific training configuration."""
    config = TransformerTrainingConfig(
        learning_rate=0.001,
        max_epochs=100,
        use_early_stopping=True,
        early_stopping_patience=10,
        transformer_labels_count=12,
        forecasting_horizon=24,
        transformer_use_teacher_forcing=True,
        transformer_use_auto_regression=False,
        attention_dropout=0.1,
        batch_size=32
    )

    # Test base attributes
    assert config.learning_rate == 0.001
    assert config.max_epochs == 100
    assert config.use_early_stopping
    assert config.early_stopping_patience == 10

    # Test transformer specific attributes
    assert config.transformer_labels_count == 12
    assert config.forecasting_horizon == 24
    assert config.transformer_use_teacher_forcing
    assert not config.transformer_use_auto_regression
    assert config.attention_dropout == 0.1
    assert config.batch_size == 32

def test_config_defaults():
    """Test configuration default values."""
    # Test base config with minimal parameters
    base_config = TrainingConfig(
        learning_rate=0.001,
        max_epochs=100,
        use_early_stopping=True,
        early_stopping_patience=10
    )
    assert base_config.batch_size == 32  # default value
    assert base_config.device == 'cuda'  # default value

    # Test neural net config with minimal parameters
    nn_config = NeuralNetTrainingConfig(
        learning_rate=0.001,
        max_epochs=100,
        use_early_stopping=True,
        early_stopping_patience=10
    )
    assert nn_config.learning_rate_scheduler_step == 30  # default value
    assert nn_config.learning_rate_scheduler_gamma == 0.1  # default value
    assert nn_config.gradient_clipping is None  # default value

    # Test transformer config with minimal parameters
    transformer_config = TransformerTrainingConfig(
        learning_rate=0.001,
        max_epochs=100,
        use_early_stopping=True,
        early_stopping_patience=10
    )
    assert transformer_config.transformer_labels_count == 1  # default value
    assert transformer_config.forecasting_horizon == 24  # default value
    assert not transformer_config.transformer_use_teacher_forcing  # default value
    assert transformer_config.attention_dropout == 0.1  # default value