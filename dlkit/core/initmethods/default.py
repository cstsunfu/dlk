
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
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_uniform_(module.weight)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight)
        elif isinstance(module, nn.Conv3d):
            torch.nn.init.kaiming_uniform_(module.weight)
