import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict, List, Set
from dlkit.core.base_module import SimpleModule, BaseModuleConfig
from . import embedding_register, embedding_config_register
from dlkit.core.modules import module_config_register, module_register

@embedding_config_register("pretrained_transformers")
class PretrainedTransformersConfig(BaseModuleConfig):
    """docstring for PretrainedTransformersConfig
    {
        "module": {
            _base: "roberta",
        },
        "config": {
            "pretrained_model_path": "*@*",
            "return_logits": {
                "embedding_logits": "embedding_logits",
                "encoder_logits": "encoder_logits",
            }, //embedding and encoder
            "input_map": {
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
                "token_type_ids": "token_type_ids",
            },
            "output_map": {
                "embedding": "embedding",
            },
            "output_size": "*@*",
        },
        "_link": {
            config.pretrained_model_path: [module.config.pretrained_model_path],
        },
        "_name": "pretrained_transformers",
    }
    """

    def __init__(self, config: Dict):
        super(PretrainedTransformersConfig, self).__init__(config)
        self.return_logits = config['config']['return_logits']
        self.pretrained_transformers_config = config["module"]
        assert self.pretrained_transformers_config['_name'] == "pretrained_transformers"
        

@embedding_register("pretrained_transformers")
class PretrainedTransformers(SimpleModule):
    def __init__(self, config: PretrainedTransformersConfig):
        super(PretrainedTransformers, self).__init__()
        self._provide_keys = {'embedding'}
        self._required_keys = {'input_ids', 'attention_mask'}
        self._provided_keys = set()
        self.config = config
        self.pretrained_transformers = module_register.get('pretrained_transformers')(module_config_register.get('pretrained_transformers')(config.pretrained_transformers_config))

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """
        """
        
        input_ids = inputs[self.get_input_name('input_ids')] if "input_ids" in self.config.input_map else None
        attention_mask = inputs[self.get_input_name('attention_mask')] if "attention_mask" in self.config.input_map else None
        token_type_ids = inputs[self.get_input_name('token_type_ids')] if "token_type_ids" in self.config.input_map else None
        token_type_ids = inputs[self.get_input_name('token_type_ids')] if "token_type_ids" in self.config.input_map else None
        inputs_embeds = inputs[self.get_input_name('inputs_embeds')] if "inputs_embeds" in self.config.input_map else None
        if (input_ids is None and inputs_embeds is None) or (input_ids is not None and inputs_embeds is not None):
            raise PermissionError("input_ids and input_embeds must set one of them to None")
        sequence_output, pooled_output, all_hidden_states, all_self_attentions = self.pretrained_transformers(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        inputs[self.get_output_name('embedding')] = sequence_output
        if self.config.return_logits.get('embedding_logits', None):
            inputs[self.config.return_logits['embedding_logits']] = all_hidden_states[0]
        if self.config.return_logits.get('encoder_logits', None):
            inputs[self.config.return_logits['encoder_logits']] = all_hidden_states[1:]
        return inputs
