"""
Colored logging utilities for CtrlFreeQ
Provides color-coded console output based on log levels with configurable colors
"""

import logging
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """
    Custom logging formatter that adds colors to log messages based on log level
    Colors are configurable via YAML configuration file
    """

    # ANSI color codes
    COLORS = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
    }

    STYLES = {
        "normal": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "underline": "\033[4m",
    }

    RESET = "\033[0m"

    def __init__(
        self,
        fmt: Optional[str] = None,
        config_path: Optional[str] = None,
        enable_colors: bool = True,
    ):
        """
        Initialize the colored formatter

        Args:
            fmt: Log message format string
            config_path: Path to color configuration YAML file
            enable_colors: Whether to enable colors (can be disabled for file output)
        """
        super().__init__(fmt)
        self.enable_colors = enable_colors
        self.color_config = self._load_color_config(config_path)

    def _load_color_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load color configuration from YAML file"""
        if config_path is None:
            # Default config path
            config_path = (
                Path(__file__).parent.parent / "config" / "logging_colors.yaml"
            )

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                return config
        except (FileNotFoundError, yaml.YAMLError):
            # Fallback to default colors if config file not found or invalid
            return {
                "colors": {
                    "DEBUG": {"color": "cyan", "style": "dim"},
                    "INFO": {"color": "white", "style": "normal"},
                    "WARNING": {"color": "yellow", "style": "bold"},
                    "ERROR": {"color": "red", "style": "bold"},
                    "CRITICAL": {"color": "magenta", "style": "bold"},
                },
                "console": {"enable_colors": True},
            }

    def _get_color_code(self, level_name: str) -> str:
        """Get ANSI color code for a log level"""
        if not self.enable_colors:
            return ""

        level_config = self.color_config.get("colors", {}).get(level_name, {})
        color = level_config.get("color", "white")
        style = level_config.get("style", "normal")

        color_code = self.COLORS.get(color, self.COLORS["white"])
        style_code = self.STYLES.get(style, self.STYLES["normal"])

        return f"{style_code}{color_code}"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors"""
        # Use different format strings based on log level
        if record.levelname == "INFO":
            # For INFO messages, show only the message (no timestamp, logger name, or level)
            formatted_message = record.getMessage()
        elif record.levelname in ["ERROR", "WARNING"]:
            # For ERROR and WARNING, show level name and message
            formatted_message = f"{record.levelname} - {record.getMessage()}"
        else:
            # For other levels (DEBUG, CRITICAL), use the full format
            formatted_message = super().format(record)

        if not self.enable_colors:
            return formatted_message

        # Get color codes for this log level
        color_start = self._get_color_code(record.levelname)
        color_end = self.RESET

        # Apply colors to the entire message
        return f"{color_start}{formatted_message}{color_end}"


def setup_colored_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    config_path: Optional[str] = None,
) -> logging.Logger:
    """
    Setup colored logging configuration for CtrlFreeQ RB framework

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs to
        config_path: Path to color configuration YAML file

    Returns:
        Configured logger instance with colored console output
    """
    logger = logging.getLogger("ctrlfreeq_rb")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers to prevent duplicates
    logger.handlers.clear()

    # Disable propagation to prevent duplicate messages from root logger
    logger.propagate = False

    # Load color configuration
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "logging_colors.yaml"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        config = {"console": {"enable_colors": True}, "file": {"enable_colors": False}}

    # Create message format
    formatter_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_colors_enabled = config.get("console", {}).get("enable_colors", True)
    console_formatter = ColoredFormatter(
        fmt=formatter_string,
        config_path=config_path,
        enable_colors=console_colors_enabled,
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler without colors (unless explicitly enabled)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_colors_enabled = config.get("file", {}).get("enable_colors", False)
        file_formatter = ColoredFormatter(
            fmt=formatter_string,
            config_path=config_path,
            enable_colors=file_colors_enabled,
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# Convenience function for backward compatibility
def setup_logging(
    level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """Backward compatible setup_logging function that now includes colors"""
    return setup_colored_logging(level=level, log_file=log_file)
