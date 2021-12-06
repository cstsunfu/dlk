import torch
from typing import Dict, List, Set, Callable
from dlkit.core.base_module import SimpleModule, BaseModuleConfig
from . import decoder_register, decoder_config_register
from dlkit.core.modules import module_config_register, module_register

@decoder_config_register("linear")
class LinearConfig(BaseModuleConfig):
    """docstring for LinearConfig
    {
        "module": {
            "_base": "linear",
        },
        "config": {
            "input_size": "*@*",
            "output_size": "*@*",
            "pool": null,
            "output_map": {},
            "input_map": {}, // required_key: provide_key
        },
        "_link":{
            "config.input_size": ["module.config.input_size"],
            "config.output_size": ["module.config.output_size"],
            "config.pool": ["module.config.pool"],
        },
        "_name": "linear",
    }
    """
    def __init__(self, config: Dict):
        super(LinearConfig, self).__init__(config)
        self.linear_config = config["module"]
        

@decoder_register("linear")
class Linear(SimpleModule):
    def __init__(self, config: LinearConfig):
        super(Linear, self).__init__(config)
        self._provide_keys = {'logits'}
        self._required_keys = {'embedding'}
        self._provided_keys = set()

        self.config = config

        self.linear = module_register.get('linear')(module_config_register.get('linear')(config.linear_config))

    def init_weight(self, method: Callable):
        """init  Module weight by `method`
        :method: init method
        :returns: None
        """
        for module in self.linear.children():
            module.apply(method)

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """
        """
        inputs[self.get_output_name("logits")] = self.linear(inputs[self.get_input_name('embedding')])
        inputs.update(self._logits_gather([inputs[self.get_output_name('logits')]]))
        return inputs
