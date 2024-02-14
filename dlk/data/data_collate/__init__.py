# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import abc
import os

from dlk.utils.import_module import import_module_dir

# automatically import any Python files in the models directory
data_collate_dir = os.path.dirname(__file__)
import_module_dir(data_collate_dir, "dlk.data.data_collate")
