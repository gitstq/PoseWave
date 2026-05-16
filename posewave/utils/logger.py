"""
Logger Utility
Configurable logging setup for PoseWave.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "posewave",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a configurable logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        format_string: Custom format string
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger("my_module", logging.DEBUG)
        >>> logger.info("Application started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class PoseWaveLogger:
    """
    Context manager for temporary logging configuration.
    
    Example:
        >>> with PoseWaveLogger("debug_module", level=logging.DEBUG):
        ...     # Debug logging enabled
        ...     pass
    """
    
    def __init__(
        self,
        name: str,
        level: int = logging.DEBUG,
        log_file: Optional[str] = None
    ):
        self.name = name
        self.level = level
        self.log_file = log_file
        self.logger = None
        self._original_level = None
    
    def __enter__(self):
        self.logger = setup_logger(self.name, self.level, self.log_file)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger:
            self.logger.setLevel(logging.INFO)
        return False
