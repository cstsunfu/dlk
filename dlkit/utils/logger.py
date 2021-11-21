import logging
import sys

_global_logger = None

def setting_logger(log_file: str):

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)

    rf_handler = logging.StreamHandler(sys.stderr)
    rf_handler.setLevel(logging.INFO)
    rf_handler.setFormatter(logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                              datefmt='%m/%d/%Y %H:%M:%S'))

    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                             datefmt='%m/%d/%Y %H:%M:%S'))

    _logger.addHandler(f_handler)

    global _global_logger
    _global_logger = _logger

def get_logger():
    """return logger if it is setted
    :returns: logger

    """
    global _global_logger
    if _global_logger:
        return _global_logger
    raise ValueError("You should setting_logger with log_file path first.")
