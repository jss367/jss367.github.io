
Having a colored logger is the most underrated thing.

You can also check out packages like these:
* https://pypi.org/project/colorlog/
* https://pypi.org/project/coloredlogs/

Let's say you're getting your logger from something like this:

logger = get_logger()


You can do this as a drop-in replacement:

from .colored_logger import get_logger



"""
Usage in other files:
from .colored_logger import get_logger
logger = get_logger(__name__)
logger.info("This is a test message")
"""

import logging
import os
import sys

from pythonjsonlogger import json

# ANSI color codes
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

# Force color output
os.environ["FORCE_COLOR"] = "1"
os.environ["TERM"] = "xterm-256color"


class ColoredJsonFormatter(json.JsonFormatter):
    def format(self, record):
        # Get the JSON string first
        json_str = super().format(record)

        # Add color based on log level
        if record.levelno >= logging.ERROR:
            color = RED
        elif record.levelno >= logging.WARNING:
            color = YELLOW
        elif record.levelno >= logging.INFO:
            color = GREEN
        else:
            color = BLUE

        # Color the entire JSON string
        return f"{color}{json_str}{RESET}"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger instance with colored JSON formatting.

    Args:
        name: The name of the logger (typically __name__)
        level: The logging level (default: logging.INFO)

    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Only add handler if it doesn't already have one
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredJsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)

    return logger


