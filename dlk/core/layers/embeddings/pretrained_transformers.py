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
from typing import Dict, List, Set
from dlk.core.base_module import SimpleModule
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules
from dlk.core.modules.bert_like import BertLikeConfig, BertLike



@config_register("embedding", 'bert_like')
@define
class BertLikeEmbeddingConfig(BertLikeConfig):
    name = NameField(value="bert_like", file=__file__, help="the static embedding module")
    @define
    class Config(BertLikeConfig.Config):
        embedding_dim = IntField(value="*@*", checker=int_check(lower=0), help="the embedding dim")
        input_map = DictField(value={
            "input_ids": "input_ids",
            "attention_mask": "attention_mask",
            "type_ids": "type_ids"
            }, help="the input map of the static embedding module")
        output_map = DictField(value={"embedding": "embedding"}, help="the output map of the static embedding module")
    config = NestField(value=Config, converter=nest_converter)


@config_register("embedding", 'bert_like@gather')
@define
class BertLikeGatherEmbeddingConfig(BertLikeConfig):
    name = NameField(value="bert_like", file=__file__, help="the static embedding module")
    @define
    class Config(BertLikeConfig.Config):
        embedding_dim = IntField(value="*@*", checker=int_check(lower=0), help="the embedding dim")
        input_map = DictField(value={
            "input_ids": "input_ids",
            "attention_mask": "subword_mask",
            "type_ids": "type_ids",
            "gather_index": "gather_index",
            }, help="the input map of the static embedding module")
        output_map = DictField(value={"embedding": "embedding"}, help="the output map of the static embedding module")
    config = NestField(value=Config, converter=nest_converter)


@register("embedding", "bert_like")
class BertLikeEmbedding(SimpleModule):
    """Wrap the hugingface transformers
    """
    def __init__(self, config: BertLikeEmbeddingConfig):
        super(BertLikeEmbedding, self).__init__(config)
        self.config = config.config
        self.pretrained_transformers = BertLike(config)

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
        input_ids = inputs[self.get_input_name('input_ids')] if "input_ids" in self.config.input_map else None
        attention_mask = inputs[self.get_input_name('attention_mask')] if "attention_mask" in self.config.input_map else None
        type_ids = inputs[self.get_input_name('type_ids')] if "type_ids" in self.config.input_map else None
        sequence_output, all_hidden_states, all_self_attentions = self.pretrained_transformers(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": type_ids,
            }
        )
        if 'gather_index' in self.config.input_map:
            gather_index = inputs[self.get_input_name("gather_index")]
            g_bs, g_seq_len = gather_index.shape
            bs, seq_len, hid_size = sequence_output.shape
            assert g_bs == bs
            assert g_seq_len <= seq_len
            sequence_output = torch.gather(sequence_output[:, :, :], 1, gather_index.unsqueeze(-1).expand(bs, g_seq_len, hid_size))
        inputs[self.get_output_name('embedding')] = sequence_output
        return inputs
