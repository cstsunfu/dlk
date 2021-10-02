"""embeddings"""
import importlib
import os
from typing import Callable, Dict, Tuple, Any
from dlkit.utils.config import Config

EMBEDDING_REGISTRY = {}
EMBEDDING_CONFIG_REGISTRY = {}

def embedding_config_register(name: str = "") -> Callable:
    """
    register configures
    """
    def decorator(config):
        if name.strip() == "":
            raise ValueError('You must set a name for {}'.format(config.__name__))

        if name in EMBEDDING_CONFIG_REGISTRY:
            raise ValueError('The embedding config name {} is already registed.'.format(name))
        EMBEDDING_CONFIG_REGISTRY[name] = config
        return config
    return decorator

def embedding_register(name: str = "") -> Callable:
    """
    register embeddings
    """
    def decorator(embedding):
        if name.strip() == "":
            raise ValueError('You must set a name for {}'.format(embedding.__name__))

        if name in EMBEDDING_REGISTRY:
            raise ValueError('The embedding name {} is already registed.'.format(name))
        EMBEDDING_REGISTRY[name] = embedding
        return embedding
    return decorator


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
import_embeddings(embeddings_dir, "embeddings")
