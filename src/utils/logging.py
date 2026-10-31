"""
Logging Configuration.
Sets up a centralized logger.
"""
import logging
import sys

DEFAULT_LOGGER_NAME = "Contours"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class SmartGroupFormatter(logging.Formatter):
    def format(self, record):
        # If the log message is our special spacer token, strip all metadata headers!
        if record.msg == "__NEW_GROUP_SPACER__":
            return ""
        return super().format(record)


def setup_logger(
        name: str = DEFAULT_LOGGER_NAME,
        level: int = logging.INFO,
        *,
        format_string: str | None = None,
        console_output: bool = True,
        file_path: str | None = None
) -> logging.Logger:
    """
    Initialize and return a configured logger.

    Parameters
    ----------
    name : str
        Name of the logger.
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO).
    format_string : str, optional
        Custom format string. Defaults to a standard pattern.
    console_output : bool
        If True, logs to stdout/stderr.
    file_path : str, optional
        If provided, logs to this file in addition to console.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
        return logger

    logger.setLevel(level)
    logger.propagate = False

    if format_string is None:
        format_string = DEFAULT_LOG_FORMAT

    formatter = SmartGroupFormatter(format_string, datefmt=DEFAULT_DATE_FORMAT)

    # Console Handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File Handler
    if file_path:
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Global default logger instance
logger = setup_logger()


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a child logger from the main logger.

    Parameters
    ----------
    name : str, optional
        Sub-module name (e.g., 'io.fits'). If None, returns the root logger.

    Returns
    -------
    logging.Logger
        The requested logger.
    """
    if name:
        child_logger = logging.getLogger(f"{DEFAULT_LOGGER_NAME}.{name}")
        # Allow child logs to bubble up to the root logger handler
        child_logger.propagate = True
        return child_logger
    return logger
