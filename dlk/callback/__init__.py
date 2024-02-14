# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict

from dlk.utils.import_module import import_module_dir

# automatically import any Python files in the models directory
callback_dir = os.path.dirname(__file__)
import_module_dir(callback_dir, "dlk.callback")
