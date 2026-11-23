"""Logging helpers for the AQI study."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with a default INFO-level configuration."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
