"""encoders"""

import importlib
import os
from dlk.utils.register import Register


encoder_config_register = Register("Encoder config register.")
encoder_register = Register("Encoder register.")

def import_encoders(encoders_dir, namespace):
    for file in os.listdir(encoders_dir):
        path = os.path.join(encoders_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            encoder_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + encoder_name)



# automatically import any Python files in the encoders directory
encoders_dir = os.path.dirname(__file__)
import_encoders(encoders_dir, "dlk.core.layers.encoders")
