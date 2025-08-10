# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os

from dlk.utils.logger import setup_logger

setup_logger(name="DLK", log_file="logs/log.txt")


from intc import cregister

from dlk.utils.register import register
