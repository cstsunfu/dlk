from transformers.models.distilbert.modeling_distilbert import DistilBertModel
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
import json
import os
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict
from . import module_register, module_config_register
from dlk.utils.config import BaseConfig

@module_config_register("distil_bert")
class DistilBertWrapConfig(BaseConfig):
    """docstring for DistilBertWrapConfig
    {
        "config": {
            "pretrained_model_path": "*@*",
            "from_pretrain": true,
            "freeze": false,
            "dropout": 0.0,
        },

        "_name": "distil_bert",
    }
    """

    def __init__(self, config: Dict):
        super(DistilBertWrapConfig, self).__init__(config)
        self.pretrained_model_path = config['config']['pretrained_model_path']
        self.from_pretrain = config['config']['from_pretrain']
        self.freeze = config['config']['freeze']
        self.dropout = config['config']['dropout']
        if os.path.isdir(self.pretrained_model_path):
            if os.path.exists(os.path.join(self.pretrained_model_path, 'config.json')):
                self.distil_bert_config = DistilBertConfig(**json.load(open(os.path.join(self.pretrained_model_path, 'config.json'), 'r')))
            else:
                raise PermissionError(f"config.json must in the dir {self.pretrained_model_path}")
        else:
            if os.path.isfile(self.pretrained_model_path):
                try:
                    self.distil_bert_config = DistilBertConfig(**json.load(open(self.pretrained_model_path, 'r')))
                except:
                    raise PermissionError(f"You must provide the pretrained model dir or the config file path.")
        self.post_check(config['config'], used=['pretrained_model_path', 'from_pretrain', 'freeze', 'dropout'])


@module_register("distil_bert")
class DistilBertWrap(nn.Module):
    def __init__(self, config: DistilBertWrapConfig):
        super(DistilBertWrap, self).__init__()
        self.config = config

        self.distil_bert = DistilBertModel(config.distil_bert_config)
        self.dropout = nn.Dropout(float(self.config.dropout))

    def init_weight(self, method):
        """TODO: Docstring for init_weight.
        :returns: TODO

        """
        if self.config.from_pretrain:
            self.from_pretrained()
        else:
            self.distil_bert.init_weights()

    def from_pretrained(self):
        """TODO: Docstring for init.
        :pretrained_model_path: TODO
        :returns: TODO
        """
        self.distil_bert = DistilBertModel.from_pretrained(self.config.pretrained_model_path)

    def forward(self, inputs):
        """
        """
        if self.config.freeze:
            self.distil_bert.eval()
            with torch.no_grad():
                outputs = self.distil_bert(
                    input_ids = inputs.get("input_ids", None),
                    attention_mask = inputs.get("attention_mask", None),
                    head_mask = inputs.get("head_mask", None),
                    inputs_embeds = inputs.get("inputs_embeds", None),
                    output_attentions = True,
                    output_hidden_states = True,
                    return_dict = False
                )
        else:
            outputs = self.distil_bert(
                input_ids = inputs.get("input_ids", None),
                attention_mask = inputs.get("attention_mask", None),
                head_mask = inputs.get("head_mask", None),
                inputs_embeds = inputs.get("inputs_embeds", None),
                output_attentions = True,
                output_hidden_states = True,
                return_dict = False
            )
        assert len(outputs) == 3, f"Please check transformers version, the len(outputs) is 3 for version == 4.12, and this version the output logistic of distil_bert is not as the same as bert and roberta."
        sequence_output, all_hidden_states, all_self_attentions = outputs[0], outputs[1], outputs[2]
        sequence_output = self.dropout(sequence_output)
        return sequence_output, all_hidden_states, all_self_attentions
