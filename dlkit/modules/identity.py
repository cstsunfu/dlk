import torch.nn as nn
from . import module_register, module_config_register
from typing import Dict, List
from dlkit.utils.base_module import SimpleModule
import torch

        
@module_config_register('identity')
class IdentityConfig(object):
    """docstring for IdentityConfig"""
    def __init__(self, config={}):
        super(IdentityConfig, self).__init__()


@module_register('identity')
class Identity(SimpleModule):
    """
    Sequence labeling model
    """

    def __init__(self, config: IdentityConfig):
        super().__init__()
        self.provide = None
        

    def provide_keys(self):
        """TODO: should provide_keys in model?
        """
        if self.provide is not None:
            return self.provide
        raise PermissionError(f"The Identity Module can only pass the inputs to outputs. so you must call check_keys_are_provided with previous module to tell Identity what should be outputs.")

    def check_keys_are_provided(self, provide: List[str])->bool:
        """TODO: should check keys in model?
        """
        self.provide = provide
        return True

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """forward
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        return inputs
