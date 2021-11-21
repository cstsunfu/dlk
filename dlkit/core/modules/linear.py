import torch.nn as nn
import torch
from typing import Dict, List
from . import module_register, module_config_register

@module_config_register("linear")
class LinearConfig(object):
    """docstring for LinearConfig
    {
        config: {
            input_size: 256,
            output_size: 2,
            dropout: 0.0, //the module output no need dropout
            bias: true, // use bias or not in linear , if set to false, all the bias will be set to 0
        },
        _name: "linear",
    }
    """
    def __init__(self, config: Dict):
        super(LinearConfig, self).__init__()
        config = config.get('config', {})
        self.input_size = config.get('input_size', 128)
        self.output_size = config.get('output_size', 128)
        self.dropout = config.get('dropout', 0.1)
        self.bias = config.get('bias', True)
        

@module_register("linear")
class Linear(nn.Module):
    def __init__(self, config: LinearConfig):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=config.input_size, out_features=config.output_size, )
        self.dropout = nn.Dropout(p=config.dropout)


    def forward(self, input: torch.Tensor)->torch.Tensor:
        """
        """
        return self.dropout(self.linear(input))
