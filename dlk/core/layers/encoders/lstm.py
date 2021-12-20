import torch.nn as nn
import torch
from typing import Dict, List, Set, Callable
from dlk.core.base_module import SimpleModule, BaseModuleConfig
from . import encoder_register, encoder_config_register
from dlk.core.modules import module_config_register, module_register
from dlk.utils.logger import Logger
logger = Logger.get_logger()

@encoder_config_register("lstm")
class LSTMConfig(BaseModuleConfig):
    """docstring for LSTMConfig
    {
        module: {
            _base: "lstm",
        },
        config: {
            input_map: {},
            output_map: {},
            input_size: *@*,
            output_size: "*@*",
            num_layers: 1,
            dropout: "*@*", // dropout between layers
        },
        _link: {
            config.input_size: [module.config.input_size],
            config.output_size: [module.config.output_size],
            config.dropout: [module.config.dropout],
        },
        _name: "lstm",
    }
    """

    def __init__(self, config: Dict):
        super(LSTMConfig, self).__init__(config)
        self.lstm_config = config["module"]
        assert self.lstm_config['_name'] == "lstm"
        self.post_check(config['config'], used=[
            "input_size",
            "output_size",
            "num_layers",
            "dropout",
        ])


@encoder_register("lstm")
class LSTM(SimpleModule):
    def __init__(self, config: LSTMConfig):
        super(LSTM, self).__init__(config)
        self._provide_keys = {'embedding'}
        self._required_keys = {'embedding', 'attention_mask'}
        self._provided_keys = set()
        self.config = config
        self.lstm = module_register.get('lstm')(module_config_register.get('lstm')(config.lstm_config))

    def init_weight(self, method: Callable):
        """init  Module weight by `method`
        :method: init method
        :returns: None
        """
        for module in self.lstm.children():
            module.apply(method)

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """
        """
        inputs[self.get_output_name('embedding')] = self.lstm(inputs[self.get_input_name('embedding')], inputs[self.get_input_name('attention_mask')])
        if self._logits_gather.layer_map:
            inputs.update(self._logits_gather([inputs[self.get_output_name('embedding')]]))
        return inputs
