import torch.nn as nn
import torch
from typing import Dict, List
from dlkit.utils.base_module import SimpleModule
from . import decoder_register, decoder_config_register
from dlkit.modules import module_config_register, module_register

@decoder_config_register("linear")
class LinearConfig(object):
    """docstring for LinearConfig
    {
        config: {
            input_size: 256,
            output_size: 2,
            dropout: 0.0, //the decoder output no need dropout
            bias: true, // use bias or not in linear , if set to false, all the bias will be set to 0
        },
        _name: "linear",
    }
    """
    def __init__(self, config: Dict):
        super(LinearConfig, self).__init__()
        self.linear_config = config
        

@decoder_register("linear")
class Linear(SimpleModule):
    def __init__(self, config: LinearConfig):
        super(Linear, self).__init__()
        self._provide_keys = ['embedding']
        self._required_keys = ['embedding']
        self._provided_keys = []

        self.linear = module_register.get('linear')(module_config_register.get('linear')(config.linear_config))

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
        return self.linear(inputs)
