# utils/logging/handlers.py
import logging
import logging.handlers
import sys
from pathlib import Path

class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Extended rotating file handler with automatic directory creation."""
    
    def __init__(self, filename: str, *args, **kwargs):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(filename, *args, **kwargs)

class ComponentHandler(logging.StreamHandler):
    """Handler that adds component information to log records."""
    
    def __init__(self, component_name: str):
        super().__init__(sys.stdout)
        self.component_name = component_name
        
    def emit(self, record: logging.LogRecord):
        record.component = self.component_name
        super().emit(record)
