from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig
import json
import os
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict
from . import module_register, module_config_register
from dlk.utils.config import BaseConfig
@module_config_register("bert")
class BertWrapConfig(BaseConfig):
    """docstring for BertWrapConfig
    {
        "config": {
            "pretrained_model_path": "*@*",
            "from_pretrain": true,
            "freeze": false,
            "dropout": 0.0,
        },
        "_name": "bert",
    }
    """

    def __init__(self, config: Dict):
        super(BertWrapConfig, self).__init__(config)
        self.pretrained_model_path = config['config']['pretrained_model_path']
        self.from_pretrain = config['config']['from_pretrain']
        self.freeze = config['config']['freeze']
        self.dropout = config['config']['dropout']
        if os.path.isdir(self.pretrained_model_path):
            if os.path.exists(os.path.join(self.pretrained_model_path, 'config.json')):
                self.bert_config = BertConfig(**json.load(open(os.path.join(self.pretrained_model_path, 'config.json'), 'r')))
            else:
                raise PermissionError(f"config.json must in the dir {self.pretrained_model_path}")
        else:
            if os.path.isfile(self.pretrained_model_path):
                try:
                    self.bert_config = BertConfig(**json.load(open(self.pretrained_model_path, 'r')))
                except:
                    raise PermissionError(f"You must provide the pretrained model dir or the config file path.")
        self.post_check(config['config'], used=['pretrained_model_path', 'from_pretrain', 'freeze', 'dropout'])


@module_register("bert")
class BertWrap(nn.Module):
    def __init__(self, config: BertWrapConfig):
        super(BertWrap, self).__init__()
        self.config = config

        self.bert = BertModel(config.bert_config, add_pooling_layer=False)
        self.dropout = nn.Dropout(float(self.config.dropout))

    def init_weight(self, method):
        """TODO: Docstring for init_weight.
        :returns: TODO

        """
        if self.config.from_pretrain:
            self.from_pretrained()
        else:
            self.bert.init_weights()

    def from_pretrained(self):
        """TODO: Docstring for init.
        :pretrained_model_path: TODO
        :returns: TODO
        """
        self.bert = BertModel.from_pretrained(self.config.pretrained_model_path)

    def forward(self, inputs):
        """
        """
        if self.config.freeze:
            with torch.no_grad():
                outputs = self.bert(
                    input_ids = inputs.get("input_ids", None),
                    attention_mask = inputs.get("attention_mask", None),
                    token_type_ids = inputs.get("token_type_ids", None),
                    position_ids = inputs.get("position_ids", None),
                    head_mask = inputs.get("head_mask", None),
                    inputs_embeds = inputs.get("inputs_embeds", None),
                    encoder_hidden_states = inputs.get("encoder_hidden_states", None),
                    encoder_attention_mask = inputs.get("encoder_attention_mask", None),
                    past_key_values = inputs.get("past_key_values", None),
                    use_cache = None,
                    output_attentions = True,
                    output_hidden_states = True,
                    return_dict = False
                )
        else:
            outputs = self.bert(
                input_ids = inputs.get("input_ids", None),
                attention_mask = inputs.get("attention_mask", None),
                token_type_ids = inputs.get("token_type_ids", None),
                position_ids = inputs.get("position_ids", None),
                head_mask = inputs.get("head_mask", None),
                inputs_embeds = inputs.get("inputs_embeds", None),
                encoder_hidden_states = inputs.get("encoder_hidden_states", None),
                encoder_attention_mask = inputs.get("encoder_attention_mask", None),
                past_key_values = inputs.get("past_key_values", None),
                use_cache = None,
                output_attentions = True,
                output_hidden_states = True,
                return_dict = False
            )
        assert len(outputs) == 4, f"Please check transformers version, the len(outputs) is 4 in version == 4.12, or check your config and remove the 'add_cross_attention'"
        sequence_output, all_hidden_states, all_self_attentions = outputs[0], outputs[2], outputs[3]
        sequence_output = self.dropout(sequence_output)
        return sequence_output, all_hidden_states, all_self_attentions
