"""decoders"""
import importlib
import os
from typing import Callable, Dict, Tuple, Any
from dlk.utils.register import Register

decoder_config_register = Register("Decoder config register.")
decoder_register = Register("Decoder register.")


def import_decoders(decoders_dir, namespace):
    for file in os.listdir(decoders_dir):
        path = os.path.join(decoders_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            decoder_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + decoder_name)


# automatically import any Python files in the decoders directory
decoders_dir = os.path.dirname(__file__)
import_decoders(decoders_dir, "dlk.core.layers.decoders")
