import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict, List, Set
from dlk.core.base_module import SimpleModule, BaseModuleConfig
from . import embedding_register, embedding_config_register
from dlk.core.modules import module_config_register, module_register

@embedding_config_register("pretrained_transformers")
class PretrainedTransformersConfig(BaseModuleConfig):
    """docstring for PretrainedTransformersConfig
    {
        "module": {
            "_base": "roberta",
        },
        "config": {
            "pretrained_model_path": "*@*",
            "input_map": {
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
                "type_ids": "type_ids",
            },
            "output_map": {
                "embedding": "embedding",
            },
            "dropout": 0, //dropout rate
            "embedding_dim": "*@*",
        },
        "_link": {
            "config.pretrained_model_path": ["module.config.pretrained_model_path"],
        },
        "_name": "pretrained_transformers",
    }
    for gather embedding
    {
        "module": {
            "_base": "roberta",
        },
        "config": {
            "pretrained_model_path": "*@*",
            "input_map": {
                "input_ids": "input_ids",
                "attention_mask": "subword_mask",
                "type_ids": "type_ids",
                "gather_index": "gather_index",
            },
            "output_map": {
                "embedding": "embedding",
            },
            "embedding_dim": "*@*",
            "dropout": 0, //dropout rate
        },
        "_link": {
            "config.pretrained_model_path": ["module.config.pretrained_model_path"],
        },
        "_name": "pretrained_transformers",
    }
    """

    def __init__(self, config: Dict):
        super(PretrainedTransformersConfig, self).__init__(config)
        self.pretrained_transformers_config = config["module"]
        self.post_check(config['config'], used=[
            "pretrained_model_path",
            "embedding_dim",
            "output_map",
            "input_map",
            "dropout",
            ])


@embedding_register("pretrained_transformers")
class PretrainedTransformers(SimpleModule):
    def __init__(self, config: PretrainedTransformersConfig):
        super(PretrainedTransformers, self).__init__(config)
        self._provide_keys = {'embedding'}
        self._required_keys = {'input_ids', 'attention_mask'}
        self.config = config
        self.pretrained_transformers = module_register.get(config.pretrained_transformers_config['_name'])(module_config_register.get(config.pretrained_transformers_config['_name'])(config.pretrained_transformers_config))

    def init_weight(self, method):
        """init  Module weight by `method`
        :method: init method, with pretrained
        :returns: None
        """
        self.pretrained_transformers.init_weight(method)

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """
        """
        input_ids = inputs[self.get_input_name('input_ids')] if "input_ids" in self.config._input_map else None
        attention_mask = inputs[self.get_input_name('attention_mask')] if "attention_mask" in self.config._input_map else None
        type_ids = inputs[self.get_input_name('type_ids')] if "type_ids" in self.config._input_map else None
        type_ids = inputs[self.get_input_name('type_ids')] if "type_ids" in self.config._input_map else None
        inputs_embeds = inputs[self.get_input_name('inputs_embeds')] if "inputs_embeds" in self.config._input_map else None
        if (input_ids is None and inputs_embeds is None) or (input_ids is not None and inputs_embeds is not None):
            raise PermissionError("input_ids and input_embeds must set one of them to None")
        sequence_output, all_hidden_states, all_self_attentions = self.pretrained_transformers(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": type_ids,
                "inputs_embeds": inputs_embeds,
            }
        )
        if 'gather_index' in self.config._input_map:
            # gather_index.shape == bs*real_sent_len
            gather_index = inputs[self.get_input_name("gather_index")]
            g_bs, g_seq_len = gather_index.shape
            bs, seq_len, hid_size = sequence_output.shape
            assert g_bs == bs
            assert g_seq_len <= seq_len
            sequence_output = torch.gather(sequence_output[:, :, :], 1, gather_index.unsqueeze(-1).expand(bs, g_seq_len, hid_size))
        inputs[self.get_output_name('embedding')] = sequence_output
        if self._logits_gather.layer_map:
            inputs.update(self._logits_gather(all_hidden_states))
        return inputs
