# utils/logging/config.py
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from typing import Optional, Union

class LogLevel(Enum):
    """Logging levels enumeration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

@dataclass
class LoggerConfig:
    """Configuration for logger instances."""
    level: LogLevel = LogLevel.INFO
    file_path: Optional[Path] = None
    format_string: Optional[str] = None
    include_timestamp: bool = True
    include_level: bool = True
    include_module: bool = True
    json_output: bool = False
    component_name: str = "main"