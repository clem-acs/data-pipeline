"""
Logging utilities for pipeline transforms.
"""

import logging
import sys
from typing import Optional


def setup_logging(base_logger_name: str = "", verbose: bool = False, log_file: Optional[str] = None):
    """Set up base logging for the pipeline (e.g., for the root logger or a main package logger).

    This function should typically be called once at the application entry point.
    Individual modules should then use logging.getLogger(__name__) and will inherit
    settings from the base logger configured here.
    
    Args:
        base_logger_name: Name of the base logger to configure (e.g., "" for the root logger,
                          or "data-pipeline" for the main application package).
        verbose: If True, set log level to DEBUG, otherwise INFO.
        log_file: Optional file path to write logs to.
    
    Returns:
        Configured base logger instance.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Get the base logger
    logger = logging.getLogger(base_logger_name)
    logger.setLevel(log_level) # Set level on this base logger
    
    # Clear any existing handlers on THIS base logger to avoid duplication 
    # if this setup function is inadvertently called multiple times for the same base_logger_name.
    # Child loggers should not have their handlers cleared by this.
    logger.handlers = [] 
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    # Set handler level - messages below this level will be ignored by THIS handler.
    console_handler.setLevel(log_level) 
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add console handler to the base logger
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level) # Set level on file handler too
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Quieten overly verbose external libraries
    # These will apply regardless of the base_logger_name because getLogger gets these specific loggers.
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('s3transfer').setLevel(logging.WARNING)
    
    # Child loggers (e.g., logging.getLogger(__name__) in other modules)
    # will inherit their effective level from this configured base logger if their own level is NOTSET.
    # Their messages will propagate to the handlers attached to this base logger.
    
    return logger