"""encoders"""

import importlib
import os
from typing import Callable, Dict, Tuple, Any
from utils.config import Config

MODEL_REGISTRY = {}
MODEL_CONFIG_REGISTRY = {}

def model_config_register(name: str = "") -> Callable:
    """
    register configures
    """
    def decorator(config):
        if name.strip() == "":
            raise ValueError('You must set a name for {}'.format(config.__name__))

        if name in MODEL_CONFIG_REGISTRY:
            raise ValueError('The model config name {} is already registed.'.format(name))
        MODEL_CONFIG_REGISTRY[name] = config
        return config
    return decorator

def model_register(name: str = "") -> Callable:
    """
    register models
    """
    def decorator(model):
        if name.strip() == "":
            raise ValueError('You must set a name for {}'.format(model.__name__))

        if name in MODEL_REGISTRY:
            raise ValueError('The model name {} is already registed.'.format(name))
        MODEL_REGISTRY[name] = model
        return model
    return decorator


def import_models(models_dir, namespace):
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + model_name)


class ModelConfig(Config):
    """docstring for ModelConfig"""
    def __init__(self, **kwargs):
        super(ModelConfig, self).__init__(**kwargs)


    def _get_sub_module(self, module_register: Dict, module_config_register: Dict, module_name: str, config: Dict) -> Tuple[Any, Config]:
        """get sub module and config from register.

        :module_register: TODO
        :module_config_register: TODO
        :module_name: TODO
        :config: Dict: TODO
        :returns: TODO

        """
        if isinstance(config, str):
            name = config
            extend_config = {}
        else:
            assert isinstance(config, dict), "{} config must be name(str) or config(dict), but you provide {}".format(module_name, config)
            for key in config:
                if key not in ['name', 'config']:
                    raise KeyError('You can only provide the {} name("name") and embedding config("config")'.format(module_name))
            name = config.get('name')
            extend_config = config.get('config', {})
            if not name:
                raise KeyError('You must provide the {} name("name")'.format(module_name))

        module, module_config =  module_register.get(name), module_config_register.get(name)
        if (not module) or not (module_config):
            raise KeyError('The {} name {} is not registed.'.format(module_name, config))
        module_config.update(extend_config)
        return module, module_config


    def get_embedding(self, config):
        """get embedding config and embedding module

        :config: TODO
        :returns: TODO

        """
        return self._get_sub_module(embedding_register, embedding_config_register, "embedding", config)
        
    def get_encoder(self, config):
        """get encoder config and encoder module

        :config: TODO
        :returns: TODO

        """
        return self._get_sub_module(encoder_register, encoder_config_register, "encoder", config)
        
    def get_decoder(self, config):
        """get decoder config and decoder module

        :config: TODO
        :returns: TODO

        """
        return self._get_sub_module(decoder_register, decoder_config_register, "decoder", config)

# automatically import any Python files in the models directory
models_dir = os.path.dirname(__file__)
import_models(models_dir, "models")
