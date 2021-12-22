# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import colorlog


class Logger(object):
    """docstring for logger"""
    global_logger = None
    global_log_file = None
    global_file_handler = None
    global_log_file = None
    color_config = {
        'DEBUG': 'white',  # cyan white
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    def __init__(self, log_file: str='', base_dir: str='logs', log_level: str='debug', log_name='dlk'):
        super(Logger, self).__init__()
        self.log_file = log_file
        self.base_dir = base_dir
        if self.base_dir and not os.path.isdir(self.base_dir):
            os.mkdir(self.base_dir)

        self.init_global_logger(log_level=log_level, log_name=log_name)
        self.init_file_logger(log_file, base_dir, log_level=log_level)

    @staticmethod
    def get_logger()->logging.Logger:
        """return the 'dlk' logger if initialized otherwise init and return it

        Returns: 
            Logger.global_logger

        """
        if Logger.global_logger is None:
            Logger.init_global_logger()
            Logger.global_logger.warning("You didn't init the logger, so we use the default logger setting to output to terminal, you can always set the file logger by yourself.")
        return Logger.global_logger

    @staticmethod
    def init_file_logger(log_file, base_dir='logs', log_level: str='debug'):
        """init(if there is not one) or change(if there already is one) the log file

        Args:
            log_file: log file path
            base_dir: real log path is '$base_dir/$log_file'
            log_level: 'debug', 'info', etc.

        Returns: 
            None

        """
        if log_file:
            log_file = os.path.join(base_dir, log_file)
        if log_file and log_file != Logger.global_log_file:
            if Logger.global_file_handler is not None:
                Logger.global_logger.removeHandler(Logger.global_file_handler)
            file_handler = logging.FileHandler(filename=log_file, mode='a', encoding='utf8')
            file_handler.setLevel(Logger.level_map[log_level])

            file_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                                                      datefmt='%m/%d/%Y %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            Logger.global_file_handler = file_handler
            Logger.global_logger.addHandler(file_handler)

    @staticmethod
    def init_global_logger(log_level: str='debug', log_name: str='dlk'):
        """init the global_logger

        Args:
            log_level: you can change this to logger to different level
            log_name: change this is not suggested 

        Returns: 
            None

        """
        if Logger.global_logger is None:
            Logger.global_logger = logging.getLogger(log_name)
            Logger.global_logger.setLevel(Logger.level_map[log_level])
            console_handler = logging.StreamHandler()
            console_handler.setLevel(Logger.level_map[log_level])
            console_formatter = colorlog.ColoredFormatter(
                fmt='%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S',
                log_colors=Logger.color_config)
            console_handler.setFormatter(console_formatter)
            Logger.global_logger.addHandler(console_handler)
