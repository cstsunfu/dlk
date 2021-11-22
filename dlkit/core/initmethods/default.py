
import torch.nn as nn
from . import initmethod_register, initmethod_config_register
from typing import Dict, List
import torch

        
# TODO: 
@initmethod_config_register('default')
class RangeNormInitConfig(object):
    """
        {
            _name: default,
            config: {
            }
        }
    """
    def __init__(self, config):
        super(RangeNormInitConfig, self).__init__()

@initmethod_register('default')
class RangeNormInit(object):
    """
    """

    def __init__(self, config: RangeNormInitConfig):
        super().__init__()

    def __call__(self, module):
        """Initialize the weights"""
        raise PermissionError("not  applied")
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=self.range)
