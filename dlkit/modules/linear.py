import torch.nn as nn
import torch
from typing import Dict, List
from dlkit.utils.base_module import SimpleModule
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
class Linear(SimpleModule):
    def __init__(self, config: LinearConfig):
        super(Linear, self).__init__()
        self._provide_keys = ['embedding']
        self._required_keys = ['embedding']
        self._provided_keys = []

        self.linear = nn.Linear(in_features=config.input_size, out_features=config.output_size, )
        self.dropout = nn.Dropout(p=config.dropout)

    def provide_keys(self):
        """TODO: should provide_keys in model?
        """
        if self.provide_keys:
            return self._provided_keys + self._provide_keys
        return self._provide_keys

    def check_keys_are_provided(self, provide: List[str])->None:
        """TODO: should check keys in model?
        """
        self._provided_keys = provide
        for required_key in self._required_keys:
            if required_key not in provide:
                raise PermissionError(f"The StaticEmbedding Module required 'input_ids' as input. You should explicit provide the provided keys (list[str]) for check.")

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """
        """
        inputs['embedding'] = self.dropout(self.linear(inputs['embedding']))
        return inputs
