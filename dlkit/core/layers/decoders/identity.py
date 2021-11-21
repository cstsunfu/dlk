import torch.nn as nn
from . import decoder_register, decoder_config_register
from typing import Dict, List
from dlkit.core.base_module import SimpleModule
import torch

        
@decoder_config_register('identity')
class IdentityDecoderConfig(object):
    """docstring for IdentityDecoderConfig"""
    def __init__(self, config):
        super(IdentityDecoderConfig, self).__init__()


@decoder_register('identity')
class IdentityDecoder(SimpleModule):
    """
    Sequence labeling model
    """

    def __init__(self, config: IdentityDecoderConfig):
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

