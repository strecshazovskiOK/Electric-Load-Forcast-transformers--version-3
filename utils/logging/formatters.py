# utils/logging/formatters.py
import json
import logging
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
            "component": getattr(record, "component", "unknown")
        }
        
        # Properly handle extra data if it exists in record.__dict__
        if "extra_data" in getattr(record, "__dict__", {}):
            log_data.update(record.__dict__["extra_data"] or {})
            
        return json.dumps(log_data)

