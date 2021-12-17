import logging
import os
import colorlog

_global_logger = None


def setting_logger(log_file: str, base_dir: str="log"):
    log_colors_config = {
        'DEBUG': 'white',  # cyan white
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }


    logger = logging.getLogger('dkit')
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    if log_file:
        if not os.path.isdir(base_dir):
            os.mkdir(base_dir)
        log_file = os.path.join(base_dir, log_file)
        file_handler = logging.FileHandler(filename=log_file, mode='a', encoding='utf8')
        file_handler.setLevel(logging.INFO)

        file_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                                                  datefmt='%m/%d/%Y %H:%M:%S')
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    console_formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        log_colors=log_colors_config)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    global _global_logger
    _global_logger = logger

def logger():
    """return logger if it is setted
    :returns: logger

    """
    global _global_logger
    if _global_logger:
        return _global_logger
    raise PermissionError("You should setting_logger with log_file path first.")
