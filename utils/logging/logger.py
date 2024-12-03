# utils/logging/logger.py
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Optional, Union

from .config import LoggerConfig, LogLevel
from .formatters import JSONFormatter
from .handlers import RotatingFileHandler, ComponentHandler

class Logger:
    """Main logger class with component-based configuration."""
    
    _instances: Dict[str, 'Logger'] = {}
    
    @classmethod
    def get_logger(cls, name: str, config: Optional[LoggerConfig] = None) -> 'Logger':
        """Get or create a logger instance."""
        if name not in cls._instances:
            cls._instances[name] = cls(name, config or LoggerConfig())
        return cls._instances[name]

    def __init__(self, name: str, config: LoggerConfig):
        """Initialize logger with configuration."""
        self.logger = logging.getLogger(name)
        self.config = config
        self.setup_logger()

    def setup_logger(self) -> None:
        """Configure logger with handlers and formatting."""
        self.logger.setLevel(self.config.level.value)
        self.logger.handlers.clear()

        # Create formatter
        if self.config.json_output:
            formatter = JSONFormatter()
        else:
            format_parts = []
            if self.config.include_timestamp:
                format_parts.append('%(asctime)s')
            if self.config.include_level:
                format_parts.append('%(levelname)s')
            if self.config.include_module:
                format_parts.append('%(module)s')
            format_parts.append('[%(component)s] %(message)s')
            
            formatter = logging.Formatter(' - '.join(format_parts))

        # Enable Windows console to process ANSI escape sequences
        if sys.platform == 'win32':
            os.system('color')

        # Configure console output with UTF-8 encoding
        if sys.platform == 'windows':
            # Enable Windows console to process ANSI escape sequences
            os.system('color')
            # Set console to UTF-8 mode
            os.system('chcp 65001')
        
        # Add console handler with proper encoding
        console_handler = logging.StreamHandler(
            open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace')
            if sys.platform == 'windows' else sys.stdout
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Add file handler if configured
        if self.config.file_path:
            file_handler = RotatingFileHandler(
                str(self.config.file_path),
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def set_level(self, level: Union[LogLevel, str]) -> None:
        """Change logging level."""
        if isinstance(level, str):
            level = LogLevel[level]
        self.config.level = level
        self.logger.setLevel(level.value)

    def debug(self, msg: str, extra: Optional[Dict] = None) -> None:
        """Log debug message."""
        log_extra = {"component": self.config.component_name}
        if extra:
            log_extra.update(extra)
        self.logger.debug(msg, extra=log_extra)

    def info(self, msg: str, extra: Optional[Dict] = None) -> None:
        """Log info message."""
        log_extra = {"component": self.config.component_name}
        if extra:
            log_extra.update(extra)
        self.logger.info(msg, extra=log_extra)


    def warning(self, msg: str, extra: Optional[Dict] = None) -> None:
        """Log warning message."""
        log_extra = {"component": self.config.component_name}
        if extra:
            log_extra.update(extra)
        self.logger.warning(msg, extra=log_extra)

    def error(self, msg: str, extra: Optional[Dict] = None) -> None:
        """Log error message."""
        log_extra = {"component": self.config.component_name}
        if extra:
            log_extra.update(extra)
        self.logger.error(msg, extra=log_extra)

    def critical(self, msg: str, extra: Optional[Dict] = None) -> None:
        """Log critical message."""
        log_extra = {"component": self.config.component_name}
        if extra:
            log_extra.update(extra)
        self.logger.critical(msg, extra=log_extra)
