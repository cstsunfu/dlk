# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os


def import_module_dir(module_dir, namespace):
    for file in os.listdir(module_dir):
        path = os.path.join(module_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            submodule_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + submodule_name)
