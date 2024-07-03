"""
Utility functions for data handling, logging, and performance tracking.

Author: 254StudioZ LLC
Date: 2024-07-01
"""

import logging

def get_logger(name):
    """
    Creates and returns a logger with the specified name.

    Parameters:
    name (str): The name of the logger.

    Returns:
    Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# Improvement: Adding a function to set log level
def set_log_level(logger, level):
    """
    Sets the logging level for the given logger.

    Parameters:
    logger (Logger): The logger instance.
    level (str): Logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').

    Returns:
    None
    """
    level = level.upper()
    if level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        logger.setLevel(getattr(logging, level))
    else:
        raise ValueError(f"Invalid log level: {level}")
