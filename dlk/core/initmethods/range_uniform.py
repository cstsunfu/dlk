import torch.nn as nn
from dlk.utils.config import BaseConfig
from . import initmethod_register, initmethod_config_register
from typing import Dict, List
import torch


@initmethod_config_register('range_uniform')
class RangeNormInitConfig(BaseConfig):
    """
        {
            "_name": "range_uniform",
            "config": {
                "range": 0.1,
            }
        }
    """
    def __init__(self, config):
        super(RangeNormInitConfig, self).__init__(config)
        range = config.get("range", 0.1)
        if isinstance(range, list):
            assert len(range) == 2
            self.range_from = range[0]
            self.range_to = range[1]
        else:
            assert isinstance(range, float)
            self.range_from = -abs(range)
            self.range_to = abs(range)
        self.post_check(config['config'], used='range')

@initmethod_register('range_uniform')
class RangeNormInit(object):
    """for transformers
    """

    def __init__(self, config: RangeNormInitConfig):
        super().__init__()
        self.config = config

    def __call__(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(from_=self.config.range_from, to=self.config.range_to)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.uniform_(from_=self.config.range_from, to=self.config.range_to)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            module.weight.data.uniform_(from_=self.config.range_from, to=self.config.range_to)
        elif isinstance(module, nn.Conv2d):
            module.weight.data.uniform_(from_=self.config.range_from, to=self.config.range_to)
        elif isinstance(module, nn.Conv3d):
            module.weight.data.uniform_(from_=self.config.range_from, to=self.config.range_to)
