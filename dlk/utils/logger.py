# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from logging.handlers import RotatingFileHandler


class CustomFormatter(logging.Formatter):
    cyan = "\x1b[36m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_rep = "%(asctime)s - dlk -%(levelname)7s -%(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_rep + reset,
        logging.INFO: cyan + format_rep + reset,
        logging.WARNING: yellow + format_rep + reset,
        logging.ERROR: red + format_rep + reset,
        logging.CRITICAL: bold_red + format_rep + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


global_logger = logging.getLogger()
log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
log_level = log_level_map.get(os.environ.get("DLK_LOG_LEVEL", "INFO"), "INFO")
global_logger.setLevel(log_level)


if not global_logger.hasHandlers():
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    ch.setFormatter(CustomFormatter())

    global_logger.addHandler(ch)
    global_logger.propagate = False


def logfile(
    file_name: str = "logs/log.log",
    max_bytes: int = 10**7,
    backup_count: int = 5,
    log_level: str = "INFO",
):
    """add file handler to logger

    Args:
        file_name (str): the file name to log
        max_bytes (int, optional): max bytes of the log file. Defaults to 10**7 bytes(10MB).
        backup_count (int, optional): backup count of the log file. Defaults to 5.
        log_level (str, optional): log level. Defaults to "INFO".
    Returns:
        None
    """
    if os.path.dirname(file_name) and not os.path.isdir(os.path.dirname(file_name)):
        os.mkdir(os.path.dirname(file_name))
    file_handler = RotatingFileHandler(
        file_name, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(log_level_map.get(log_level, "INFO"))
    formatter = logging.Formatter(
        "%(asctime)s - dlk - %(levelname)8s - %(filename)8s:%(lineno)4d - %(message)s"
    )
    file_handler.setFormatter(formatter)
    global_logger.addHandler(file_handler)
