"""callbacks"""

import importlib
import os
from typing import Dict, Any
from dlk.utils.register import Register

callback_config_register = Register("Callback config register")
callback_register = Register("Callback register")


def import_callbacks(callbacks_dir, namespace):
    for file in os.listdir(callbacks_dir):
        path = os.path.join(callbacks_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            callback_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + callback_name)


# automatically import any Python files in the models directory
callbacks_dir = os.path.dirname(__file__)
import_callbacks(callbacks_dir, "dlk.core.callbacks")
