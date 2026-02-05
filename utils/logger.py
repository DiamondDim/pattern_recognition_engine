"""
Logging configuration for Pattern Recognition Engine.
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional

import config


def setup_logging(log_level: Optional[str] = None,
                  log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file

    Returns:
        Configured logger instance
    """
    # Use config values if not provided
    if log_level is None:
        log_level = config.LOG_LEVEL
    if log_file is None:
        log_file = config.LOG_FILE

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create log directory {log_dir}: {e}")
            # Fallback to current directory
            log_file = "pattern_engine.log"

    # Configure root logger
    logger = logging.getLogger()

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set log level
    log_level_numeric = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(log_level_numeric)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler (detailed)
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")

    # Console handler (simple)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level_numeric)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # Set specific log levels for noisy libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)

    # Log startup message
    logger.info(f"Logging configured. Level: {log_level}, File: {log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggingMixin:
    """Mixin class to add logging capabilities to other classes."""

    def __init__(self, *args, **kwargs):
        """Initialize the mixin."""
        super().__init__(*args, **kwargs)
        self._logger = None

    @property
    def logger(self) -> logging.Logger:
        """Get the logger for this class."""
        if self._logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger

    def log_debug(self, message: str, *args, **kwargs):
        """Log a debug message."""
        self.logger.debug(message, *args, **kwargs)

    def log_info(self, message: str, *args, **kwargs):
        """Log an info message."""
        self.logger.info(message, *args, **kwargs)

    def log_warning(self, message: str, *args, **kwargs):
        """Log a warning message."""
        self.logger.warning(message, *args, **kwargs)

    def log_error(self, message: str, *args, **kwargs):
        """Log an error message."""
        self.logger.error(message, *args, **kwargs)

    def log_exception(self, message: str, *args, **kwargs):
        """Log an exception with traceback."""
        self.logger.exception(message, *args, **kwargs)


# Create default logger
logger = setup_logging()

