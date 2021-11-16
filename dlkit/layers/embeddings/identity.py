import torch.nn as nn
from . import embedding_register, embedding_config_register
from typing import Dict, List
from dlkit.utils.base_module import SimpleModule
import torch

        
@embedding_config_register('identity')
class IdentityEmbeddingConfig(object):
    """docstring for BasicModelConfig"""
    def __init__(self, config):
        super(IdentityEmbeddingConfig, self).__init__()


@embedding_register('identity')
class IdentityEmbedding(SimpleModule):
    """
    Sequence labeling model
    """

    def __init__(self, config: IdentityEmbeddingConfig):
        super().__init__()
        self._provided_keys = None
        
    def provide_keys(self):
        """TODO: should provide_keys in model?
        """
        if self._provided_keys is not None:
            return self._provided_keys
        raise PermissionError(f"The Identity Module can only pass the inputs to outputs. so you must call check_keys_are_provided with previous module to tell Identity what should be outputs.")

    def check_keys_are_provided(self, provide: List[str])->None:
        """TODO: should check keys in model?
        """
        self._provided_keys = provide

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """forward
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        return inputs
