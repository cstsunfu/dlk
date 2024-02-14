# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Dict, Tuple

from dlk.utils.import_module import import_module_dir

initmethod_dir = os.path.dirname(__file__)
import_module_dir(initmethod_dir, "dlk.initmethod")
