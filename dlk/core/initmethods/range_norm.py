import torch.nn as nn
from dlk.utils.config import BaseConfig
from . import initmethod_register, initmethod_config_register
from typing import Dict, List
import torch


@initmethod_config_register('range_norm')
class RangeNormInitConfig(BaseConfig):
    """
        {
            "_name": "range_norm",
            "config": {
                "range": 0.1,
            }
        }
    """
    def __init__(self, config):
        super(RangeNormInitConfig, self).__init__(config)
        self.range = config.get("range", 0.1)
        self.post_check(config['config'], used=['range'])

@initmethod_register('range_norm')
class RangeNormInit(object):
    """for transformers
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
        elif isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=self.range)
        elif isinstance(module, nn.Conv3d):
            module.weight.data.normal_(mean=0.0, std=self.range)
