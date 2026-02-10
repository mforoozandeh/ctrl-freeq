"""
Universal logging configuration for CtrlFreeQ
Automatically configures colored logging for the entire repository without requiring explicit setup
"""

import logging
import os
from typing import Optional

# Import colored logging utilities
try:
    from .colored_logging import setup_colored_logging
except ImportError:
    # Fallback if colored logging not available
    def setup_colored_logging(level: str = "INFO", log_file: Optional[str] = None):
        logger = logging.getLogger("ctrlfreeq_rb")
        logger.setLevel(getattr(logging, level.upper()))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger


# Global configuration
_UNIVERSAL_LOGGER_INITIALIZED = False
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_LOG_FILE = None


def get_default_log_level() -> str:
    """Get default log level from environment variable or config"""
    return os.environ.get(
        "CTRLFREEQ_LOG_LEVEL", os.environ.get("CtrlFreeQ_LOG_LEVEL", _DEFAULT_LOG_LEVEL)
    )


def get_default_log_file() -> Optional[str]:
    """Get default log file from environment variable or config"""
    return os.environ.get(
        "CTRLFREEQ_LOG_FILE", os.environ.get("CtrlFreeQ_LOG_FILE", _DEFAULT_LOG_FILE)
    )


def initialize_universal_logging() -> logging.Logger:
    """
    Initialize universal logging for CtrlFreeQ if not already initialized

    This function is called automatically when any CtrlFreeQ module is imported.
    It sets up colored logging with sensible defaults that can be overridden
    via environment variables.

    Environment Variables:
        CTRLFREEQ_LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR)
        CTRLFREEQ_LOG_FILE: Set log file path (optional)

    Returns:
        Configured logger instance
    """
    global _UNIVERSAL_LOGGER_INITIALIZED

    if _UNIVERSAL_LOGGER_INITIALIZED:
        return logging.getLogger("ctrlfreeq_rb")

    # Get configuration from environment or defaults
    log_level = get_default_log_level()
    log_file = get_default_log_file()

    # Initialize colored logging
    logger = setup_colored_logging(level=log_level, log_file=log_file)

    # Ensure no propagation to prevent duplicate messages
    logger.propagate = False

    # Mark as initialized
    _UNIVERSAL_LOGGER_INITIALIZED = True

    # Log initialization (only at DEBUG level to avoid noise)
    if log_level.upper() == "DEBUG":
        logger.debug(f"CtrlFreeQ universal logging initialized with level: {log_level}")
        if log_file:
            logger.debug(f"CtrlFreeQ logging to file: {log_file}")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with universal configuration

    Args:
        name: Logger name (defaults to "ctrlfreeq_rb")

    Returns:
        Configured logger instance
    """
    # Ensure universal logging is initialized
    initialize_universal_logging()

    # Return the appropriate logger
    if name is None:
        return logging.getLogger("ctrlfreeq_rb")
    else:
        return logging.getLogger(name)


# Initialize universal logging when this module is imported
_universal_logger = initialize_universal_logging()

# Export the universal logger for backward compatibility
logger = _universal_logger
