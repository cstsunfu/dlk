
import torch.nn as nn
from . import initmethod_register, initmethod_config_register
from typing import Dict, List
import torch

        
@initmethod_config_register('range_norm')
class RangeNormInitConfig(object):
    """
        {
            _name: range_norm,
            config: {
                range: 0.01,
            }
        }
    """
    def __init__(self, config):
        super(RangeNormInitConfig, self).__init__()
        self.range = config.get("range", 0.01)

@initmethod_register('range_norm')
class RangeNormInit(object):
    """
    """

    def __init__(self, config: RangeNormInitConfig):
        super().__init__()
        self.range = config.range
        

    def __call__(self, module):
        """Initialize the weights"""
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
