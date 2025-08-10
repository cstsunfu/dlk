# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

"""A centralized and intelligent logging setup module.

This module provides a single function, `setup_logger`, to configure the root
logger for an application. It automatically detects distributed environments
like PyTorch DDP and Ray to restrict console output to the main rank (rank 0),
while allowing flexible file logging for all ranks.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


class CustomFormatter(logging.Formatter):
    """A custom logging formatter that adds color to console output."""

    cyan = "\x1b[36m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, name: str = "app"):
        """Initializes the formatter."""
        super().__init__()
        format_rep = f"%(asctime)s - {name} - %(levelname)7s - %(message)s"
        self.FORMATS = {
            logging.DEBUG: self.grey + format_rep + self.reset,
            logging.INFO: self.cyan + format_rep + self.reset,
            logging.WARNING: self.yellow + format_rep + self.reset,
            logging.ERROR: self.red + format_rep + self.reset,
            logging.CRITICAL: self.bold_red + format_rep + self.reset,
        }

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record with level-specific colors.

        Args:
            record: The log record to format.

        Returns:
            The formatted, colorized log string.
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class RankFilter(logging.Filter):
    """A logging filter that only allows records from a specific rank."""

    def __init__(self, rank: int = 0, ray_tune=True):
        """Initializes the filter.

        Args:
            rank: The integer rank to allow logs from.
        """
        super().__init__()
        self.rank_to_allow = rank
        self.ray_tune = ray_tune

    def filter(self, record: logging.LogRecord) -> bool:
        """Determines if a log record should be processed.

        Args:
            record: The log record to check.

        Returns:
            True if the record's process rank matches the allowed rank,
            False otherwise.
        """
        local_rank = os.environ.get("LOCAL_RANK", "0")
        is_in_tune_trial = os.environ.get("TUNE_ORIG_WORKING_DIR") is not None
        if (is_in_tune_trial and self.ray_tune) or local_rank != "0":
            return False
        return True


# --- Main Setup Function ---

_LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

root_logger = logging.getLogger()


def setup_logger(
    name: str = "DLK",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_file_per_rank: Optional[bool] = None,
    max_bytes: int = 10**7,
    backup_count: int = 5,
):
    """Initializes a global logger with smart distributed environment handling.

    This function is idempotent; it will not add handlers if the root logger
    is already configured. In distributed environments, it automatically
    restricts console output to rank 0.

    Args:
        name: The name of the application, used in log messages.
        log_level: The minimum log level to process. Can be overridden by the
            `APP_LOG_LEVEL` environment variable.
        log_file: If provided, a `RotatingFileHandler` will be added to log
            to this file path.
        max_bytes: The maximum size of a log file before rotation.
        backup_count: The number of backup log files to keep.
    """
    # clean logger if handlers already exist.
    root_logger.handlers = []

    # Determine log level from argument or environment variable.
    level_str = os.environ.get("DLK_LOG_LEVEL", log_level).upper()
    level = _LOG_LEVEL_MAP.get(level_str, logging.INFO)
    root_logger.setLevel(level)

    # Configure console handler with automatic rank filtering.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter(name=name))

    console_handler.addFilter(RankFilter(rank=0))
    root_logger.addHandler(console_handler)

    # Configure file handler if a path is provided.
    if log_file:
        final_log_path = log_file
        if log_file_per_rank is None:
            env_val = os.environ.get("LOG_FILE_PER_RANK", "false").lower()
            should_log_per_rank = env_val in ("true", "1", "t", "y", "yes")
        else:
            should_log_per_rank = log_file_per_rank

        try:
            log_dir = os.path.dirname(final_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = RotatingFileHandler(
                final_log_path, maxBytes=max_bytes, backupCount=backup_count
            )
            # Use a more detailed format for file logs.
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - Rank:%(process)d - %(levelname)8s - "
                "%(filename)s:%(lineno)4d - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            if not should_log_per_rank:
                file_handler.addFilter(RankFilter(rank=0, ray_tune=False))
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.error(f"Failed to add file handler for '{final_log_path}': {e}")

    # Prevent logs from propagating to the parent logger (the default root).
    root_logger.propagate = False


def change_log_file(new_log_path: str):
    """Dynamically changes the output file of the existing file handler.

    This function finds the first active `RotatingFileHandler` on the root
    logger, closes its current file stream, and points it to a new file path.
    The directory for the new path will be created if it doesn't exist.

    If no `RotatingFileHandler` is found, a warning is logged.

    Args:
        new_log_path: The full path to the new log file.
    """
    target_handler = None
    # Find the first RotatingFileHandler instance
    for handler in root_logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            target_handler = handler
            break

    if target_handler:
        # Ensure the new directory exists
        new_log_dir = os.path.dirname(new_log_path)
        if new_log_dir:
            os.makedirs(new_log_dir, exist_ok=True)

        old_path = target_handler.baseFilename

        # This is the crucial part:
        # 1. Close the current file stream.
        target_handler.close()
        # 2. Set the new file name.
        target_handler.baseFilename = new_log_path

        # The next log message will automatically open the new file.
        logging.info(f"Log file has been changed from '{old_path}' to '{new_log_path}'")
    else:
        logging.warning(
            f"Could not change log file to '{new_log_path}'. "
            "No RotatingFileHandler found on the root logger."
        )
