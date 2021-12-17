"""embeddings"""
import importlib
import os
from typing import Callable, Dict, Tuple, Any
from dlk.utils.register import Register

embedding_config_register = Register("Embedding config register.")
embedding_register = Register("Embedding register.")


def import_embeddings(embeddings_dir, namespace):
    for file in os.listdir(embeddings_dir):
        path = os.path.join(embeddings_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            embedding_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + embedding_name)


class EmbeddingInput(object):
    """docstring for EmbeddingInput"""
    def __init__(self, **args):
        super(EmbeddingInput, self).__init__()
        self.input_ids = args.get("input_ids", None)


class EmbeddingOutput(object):
    """docstring for EmbeddingOutput"""
    def __init__(self, **args):
        super(EmbeddingOutput, self).__init__()
        self.represent = args.get("represent", None)


# automatically import any Python files in the embeddings directory
embeddings_dir = os.path.dirname(__file__)
import_embeddings(embeddings_dir, "dlk.core.layers.embeddings")
