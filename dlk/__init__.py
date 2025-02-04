# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os

from dlk.utils.logger import logfile

if os.environ.get("DISABLE_LOGFILE", "0") not in {
    "1",
    "True",
    "true",
    "TRUE",
    "YES",
    "yes",
    "Yes",
} and os.environ.get("IN_INTC", "0") not in {"1", "True", "true", "TRUE", 1}:
    logfile()


from intc import cregister

from dlk.utils.register import register
