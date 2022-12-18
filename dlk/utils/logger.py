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

from loguru import logger as logging
from loguru._logger import Logger as LoggerClass
from typing import Set
import sys
import os


class Logger(object):
    """docstring for logger"""
    global_logger: LoggerClass = None
    global_log_file: Set[str] = set()
    log_name: str = "dlk"
    warning_file = True

    level_map = {
        "debug": "DEBUG",
        "info": "INFO",
        "warning": "WARNING",
        "error": "ERROR",
    }

    def __init__(self, log_file: str='', base_dir: str='logs', log_level: str='debug', log_name="dlk"):
        super(Logger, self).__init__()
        self.log_file = log_file
        self.base_dir = base_dir
        Logger.log_name = log_name
        if self.base_dir and not os.path.isdir(self.base_dir):
            os.mkdir(self.base_dir)

        self.init_global_logger(log_level=log_level, log_name=Logger.log_name, reinit=True)
        if self.log_file:
            self.init_file_logger(log_file, base_dir, log_level=log_level)

    @staticmethod
    def get_logger()->LoggerClass:
        """return the 'dlk' logger if initialized otherwise init and return it

        Returns: 
            Logger.global_logger

        """
        if Logger.global_logger is None:
            Logger.init_global_logger()
        if not Logger.global_log_file and Logger.warning_file:
            Logger.global_logger.warning("You didn't add the logger file, only stdout(stderr) is working.")
            Logger.warning_file = False
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
        if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] not in [-1, 0]:
            return 
        if "GLOBAL_RANK" in os.environ and os.environ["GLOBAL_RANK"] != 0:
            return 
        if log_file:
            log_file = os.path.join(base_dir, log_file)
        if Logger.global_log_file:
            Logger.global_logger.warning(f"Exists a file handler at '{'; '.join(Logger.global_log_file)}'")
        Logger.global_log_file.add(log_file)
        Logger.global_logger.add(log_file, rotation="10 MB", format="{time:MM/DD/YYYY HH:mm:ss} - {level:<8} - "+Logger.log_name+" - {message}")

    @staticmethod
    def init_global_logger(log_level: str='debug', log_name: str=None, reinit: bool=False):
        """init the global_logger

        Args:
            log_level: you can change this to logger to different level
            log_name: change this is not suggested 
            reinit: if set true, will force reinit

        Returns: 
            None

        """
        if log_name and (log_name != Logger.log_name):
            Logger.log_name = log_name
        if reinit or not Logger.global_logger:
            Logger.global_logger = logging
            Logger.global_logger.remove()

            if ("LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] not in [-1, 0]) or ("GLOBAL_RANK" in os.environ and os.environ["GLOBAL_RANK"] != 0):
                Logger.global_logger.add(sys.stdout, level=Logger.level_map["error"], format="<level>{time:MM/DD/YYYY HH:mm:ss} - {level:<8}</level> - <cyan>"+Logger.log_name+"</cyan> - <level>{message}</level>")
            else:
                Logger.global_logger.add(sys.stdout, level=Logger.level_map[log_level], format="<level>{time:MM/DD/YYYY HH:mm:ss} - {level:<8}</level> - <cyan>"+Logger.log_name+"</cyan> - <level>{message}</level>")
            # Logger.global_logger.level("INFO",  color="<g>")

