# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os

from dlk.utils.import_module import import_module_dir

# automatically import any Python files in the models directory
processor_dir = os.path.dirname(__file__)
import_module_dir(processor_dir, "dlk.data.processor")
