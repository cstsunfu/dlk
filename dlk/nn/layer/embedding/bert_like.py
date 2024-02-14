# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Set

import torch
import torch.nn as nn
from intc import (
    MISSING,
    AnyField,
    Base,
    BoolField,
    DictField,
    FloatField,
    IntField,
    ListField,
    NestField,
    StrField,
    SubModule,
    cregister,
)

from dlk.nn.base_module import SimpleModule
from dlk.nn.module.bert_like import BertLike, BertLikeConfig
from dlk.utils.register import register


@cregister("embedding", "bert_like")
class BertLikeEmbeddingConfig(BertLikeConfig):
    """
    the bert like embedding module
    """

    embedding_dim = IntField(value=MISSING, minimum=1, help="the embedding dim")

    class InputMap:
        input_ids = StrField(value="input_ids", help="the input ids")
        attention_mask = StrField(value="attention_mask", help="the attention mask")
        type_ids = StrField(value="type_ids", help="the type ids")
        gather_index = StrField(
            value="",
            help='if the gather_index is not "", we will gather the output based gather index',
        )

    input_map = NestField(
        value=InputMap, help="the input map of the static embedding module"
    )

    class OutputMap:
        embedding = StrField(value="embedding", help="the embedding")

    output_map = NestField(
        value=OutputMap, help="the output map of the static embedding module"
    )


@register("embedding", "bert_like")
class BertLikeEmbedding(SimpleModule):
    """Wrap the hugingface transformers"""

    def __init__(self, config: BertLikeEmbeddingConfig):
        super(BertLikeEmbedding, self).__init__(config)
        self.config = config
        self.pretrained_transformers = BertLike(config)

    def init_weight(self, method):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        self.pretrained_transformers.init_weight(method)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """get the transformers output as embedding

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        input_ids = (
            inputs[self.config.input_map.input_ids]
            if self.config.input_map.input_ids
            else None
        )
        attention_mask = (
            inputs[self.config.input_map.attention_mask]
            if self.config.input_map.attention_mask
            else None
        )
        type_ids = (
            inputs[self.config.input_map.type_ids]
            if self.config.input_map.type_ids
            else None
        )
        (
            sequence_output,
            all_hidden_states,
            all_self_attentions,
        ) = self.pretrained_transformers(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": type_ids,
            }
        )
        if self.config.input_map.gather_index:
            gather_index = inputs[self.config.input_map.gather_index]
            g_bs, g_seq_len = gather_index.shape
            bs, seq_len, hid_size = sequence_output.shape
            assert g_bs == bs
            assert g_seq_len <= seq_len
            sequence_output = torch.gather(
                sequence_output[:, :, :],
                1,
                gather_index.unsqueeze(-1).expand(bs, g_seq_len, hid_size),
            )
        inputs[self.config.output_map.embedding] = sequence_output
        return inputs
