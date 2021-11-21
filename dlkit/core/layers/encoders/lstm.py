import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict, List
from dlkit.core.base_module import SimpleModule
from . import encoder_register, encoder_config_register
from dlkit.core.modules import module_config_register, module_register

@encoder_config_register("lstm")
class LSTMConfig(object):
    """docstring for LSTMConfig
    {
        module: {
            config:{
                bidirectional: true,
                hidden_size: 200, //the output is 2*hidden_size if use
                input_size: 200,
                proj_size: 200,
                num_layers: 1,
                dropout: 0.1, // dropout between layers
                dropout_last: true, //dropout the last layer output or not
            }
            _base: "lstm",
        },
        config: {
            return_logits: "encoder_logits",
            output_map: {}
        }
        _name: "lstm",
    }
    """

    def __init__(self, config: Dict):
        super(LSTMConfig, self).__init__()
        self.return_logits = config.get('config', {}).get('return_logits', None)
        self.lstm_config = config.get("module", {})
        assert self.lstm_config['_name'] == "lstm"
        

@encoder_register("lstm")
class LSTM(SimpleModule):
    def __init__(self, config: LSTMConfig):
        super(LSTM, self).__init__()
        self._provide_keys = ['embedding']
        self._required_keys = ['embedding', 'attention_mask']
        self._provided_keys = []
        self.config = config
        self.lstm = module_register.get('lstm')(module_config_register.get('lstm')(config.lstm_config))

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
        inputs['embedding'] = self.lstm(inputs['embedding'], inputs['attention_mask'])
        if self.config.return_logits:
            inputs[self.config.return_logits] = inputs['embedding']
        return inputs
