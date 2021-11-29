import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict, List, Set
from dlkit.core.base_module import SimpleModule, BaseModuleConfig
from . import encoder_register, encoder_config_register
from dlkit.core.modules import module_config_register, module_register

@encoder_config_register("pretrained_transformers")
class PretrainedTransformersConfig(BaseModuleConfig):
    """docstring for PretrainedTransformersConfig
    {
        module: {
            _base: "roberta",
        },
        config: {
            return_logits: "encoder_logits",
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
        _name: "pretrained_transformers",
    }
    """

    def __init__(self, config: Dict):
        super(PretrainedTransformersConfig, self).__init__(config)
        self.return_logits = config['config']['return_logits']
        self.pretrained_transformers_config = config["module"]
        assert self.pretrained_transformers_config['_name'] == "pretrained_transformers"
        

@encoder_register("pretrained_transformers")
class PretrainedTransformers(SimpleModule):
    def __init__(self, config: PretrainedTransformersConfig):
        super(PretrainedTransformers, self).__init__()
        self._provide_keys = {'embedding'}
        self._required_keys = {'embedding', 'attention_mask'}
        self._provided_keys = set()
        self.config = config
        self.pretrained_transformers = module_register.get('pretrained_transformers')(module_config_register.get('pretrained_transformers')(config.pretrained_transformers_config))
        self.i = 0

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """
        """
        inputs[self.get_output_name('embedding')] = self.pretrained_transformers(inputs[self.get_input_name('embedding')], inputs[self.get_input_name('attention_mask')])
        if self.config.return_logits:
            inputs[self.config.return_logits] = inputs[self.get_output_name('embedding')]
        return inputs
