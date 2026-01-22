# utils/logger.py
"""
Centralized logging configuration for the Multi-Agent Research Assistant.

Features:
- Console output with Rich formatting
- File logging with rotation
- Structured log format with timestamps
- Different log levels for development vs production
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
from rich.console import Console

from utils.config import get_settings

settings = get_settings()

# Create logs directory
LOGS_DIR = Path("./logs")
LOGS_DIR.mkdir(exist_ok=True)

# Log format for file output
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "research_assistant",
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up and return a configured logger.
    
    Args:
        name: Logger name (usually module __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console with Rich formatting
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    logger.propagate = False
    
    # Console handler with Rich formatting
    if log_to_console:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True
        )
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_to_file:
        log_file = LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance. Creates one if it doesn't exist.
    
    Args:
        name: Logger name. If None, uses 'research_assistant'
        
    Returns:
        Logger instance
    """
    if name is None:
        name = "research_assistant"
    return setup_logger(name)


# Pre-configured loggers for different components
def get_agent_logger(agent_name: str) -> logging.Logger:
    """Get a logger for an agent."""
    return get_logger(f"agent.{agent_name}")


def get_util_logger(util_name: str) -> logging.Logger:
    """Get a logger for a utility module."""
    return get_logger(f"util.{util_name}")


# Quick access to main logger
logger = get_logger("research_assistant")
