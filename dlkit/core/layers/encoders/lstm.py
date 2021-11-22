import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict, List, Set
from dlkit.core.base_module import SimpleModule
from . import encoder_register, encoder_config_register
from dlkit.core.modules import module_config_register, module_register

@encoder_config_register("lstm")
class LSTMConfig(object):
    """docstring for LSTMConfig
    {
        module: {
            _base: "lstm",
        },
        config: {
            return_logits: "encoder_logits",
            output_map: {},
            hidden_size: "*@*",
            input_size: *@*,
            output_size: "*@*",
            num_layers: 1,
            dropout: "*@*", // dropout between layers
        },
        _link: {
            config.hidden_size: [module.config.hidden_size],
            config.input_size: [module.config.input_size],
            config.output_size: [module.config.proj_size],
            config.dropout: [module.config.dropout],
        },
        _name: "lstm",
    }
    """

    def __init__(self, config: Dict):
        super(LSTMConfig, self).__init__()
        self.return_logits = config['config']['return_logits']
        self.lstm_config = config["module"]
        self.output_map = config['config']['output_map']
        assert self.lstm_config['_name'] == "lstm"
        

@encoder_register("lstm")
class LSTM(SimpleModule):
    def __init__(self, config: LSTMConfig):
        super(LSTM, self).__init__()
        self._provide_keys = {'embedding'}
        self._required_keys = {'embedding', 'attention_mask'}
        self._provided_keys = set()
        self.config = config
        self.lstm = module_register.get('lstm')(module_config_register.get('lstm')(config.lstm_config))
        self.i = 0

    def provide_keys(self)->Set:
        """TODO: should provide_keys in model?
        """
        return self.set_rename(self._provided_keys.union(self._provide_keys), self.config.output_map)

    def check_keys_are_provided(self, provide: Set[str])->None:
        """
        """
        self._provided_keys = provide
        for required_key in self._required_keys:
            if required_key not in provide:
                raise PermissionError(f"The {self.__class__.__name__} Module required '{required_key}' as input.")

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """
        """
        inputs['embedding'] = self.lstm(inputs['embedding'], inputs['attention_mask'])
        if self.config.return_logits:
            inputs[self.config.return_logits] = inputs['embedding']
        return self.dict_rename(inputs, self.config.output_map)
