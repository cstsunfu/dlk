# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict, List, Set
from dlk.core.base_module import SimpleModule, BaseModuleConfig
from . import embedding_register, embedding_config_register
from dlk.core.modules import module_config_register, module_register

@embedding_config_register("pretrained_transformers")
class PretrainedTransformersConfig(BaseModuleConfig):
    default_config = {
        "_name": "pretrained_transformers",
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
            "embedding_dim": "*@*",
            "dropout": "*@*",
        },
        "_link": {
            "config.pretrained_model_path": ["module.config.pretrained_model_path"],
            "config.dropout": ["module.config.dropout"],
        },
        "module": {
            "_base": "roberta",
        },
    }
    """Config for PretrainedTransformers

    Config Example:
        default_config
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
            "return_logits",
            ])


@embedding_register("pretrained_transformers")
class PretrainedTransformers(SimpleModule):
    """Wrap the hugingface transformers
    """
    def __init__(self, config: PretrainedTransformersConfig):
        super(PretrainedTransformers, self).__init__(config)
        self._provide_keys = {'embedding'}
        self._required_keys = {'input_ids', 'attention_mask'}
        self.config = config
        self.pretrained_transformers = module_register.get(config.pretrained_transformers_config['_name'])(module_config_register.get(config.pretrained_transformers_config['_name'])(config.pretrained_transformers_config))

    def init_weight(self, method):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns: 
            None

        """
        self.pretrained_transformers.init_weight(method)

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """get the transformers output as embedding

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        input_ids = inputs[self.get_input_name('input_ids')] if "input_ids" in self.config._input_map else None
        attention_mask = inputs[self.get_input_name('attention_mask')] if "attention_mask" in self.config._input_map else None
        type_ids = inputs[self.get_input_name('type_ids')] if "type_ids" in self.config._input_map else None
        type_ids = inputs[self.get_input_name('type_ids')] if "type_ids" in self.config._input_map else None
        inputs_embeds = inputs[self.get_input_name('input_embedding')] if "input_embedding" in self.config._input_map else None
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
