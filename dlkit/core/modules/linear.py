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
            pool: null, // pooling output or not
        },
        _name: "linear",
    }
    """
    def __init__(self, config: Dict):
        super(LinearConfig, self).__init__()
        config = config['config']
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.dropout = config['dropout']
        self.bias = config['bias']
        self.pool = config['pool']
        

@module_register("linear")
class Linear(nn.Module):
    def __init__(self, config: LinearConfig):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=config.input_size, out_features=config.output_size, )
        self.dropout = nn.Dropout(p=config.dropout)
        self.config = config


    def forward(self, input: torch.Tensor)->torch.Tensor:
        """
        """
        output = self.dropout(self.linear(input))
        if not self.config.pool:
            return output
        elif self.config.pool == 'first':
            return output[:, 0]
        else:
            raise PermissionError(f"Currenttly we have not support the pool method '{self.config.pool}' in linear.")
