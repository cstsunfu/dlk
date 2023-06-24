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
from typing import Dict
from dlk.utils.config import BaseConfig
from . import Module
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules

@config_register("module", 'lstm')
@define
class LSTMConfig(BaseConfig):
    name = NameField(value="lstm", file=__file__, help="the lstm config")
    @define
    class Config:
        bidirectional = BoolField(value=True, help="whether to use bidirectional lstm")
        input_size = IntField(value="*@*", checker=int_check(lower=0), help="the input size")
        output_size = IntField(value="*@*", checker=int_check(lower=0), help="the output size")
        num_layers = IntField(value=1, checker=int_check(lower=1), help="the number of layers")
        dropout = FloatField(value=0.0, checker=float_check(lower=0.0), help="the dropout rate")
        dropout_last = BoolField(value=True, help="whether to dropout the last layer output")

    config = NestField(value=Config, converter=nest_converter)


@register("module", "lstm")
class LSTM(Module):
    "A wrap for nn.LSTM"
    def __init__(self, config: LSTMConfig):
        super(LSTM, self).__init__()
        self.config = config.config

        if self.config.num_layers <= 1:
            inlstm_dropout = 0
        else:
            inlstm_dropout = self.config.dropout

        hidden_size = self.config.output_size
        if self.config.bidirectional:
            assert self.config.output_size % 2 == 0
            hidden_size = self.config.output_size // 2

        self.lstm = nn.LSTM(input_size=self.config.input_size, hidden_size=hidden_size, num_layers=self.config.num_layers, batch_first=True, bidirectional=self.config.bidirectional, dropout=inlstm_dropout)
        self.dropout_last = nn.Dropout(p=float(self.config.dropout) if self.config.dropout_last else 0)

    def forward(self, input: torch.Tensor, mask: torch.Tensor)->torch.Tensor:
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns: 
            lstm output the shape is the same as input

        """
        max_seq_len = input.size(1)
        seq_lens = mask.sum(1).cpu()
        pack_seq_rep = pack_padded_sequence(input=input, lengths=seq_lens, batch_first=True, enforce_sorted=False)
        pack_seq_rep = self.lstm(pack_seq_rep)[0]
        output, _ = pad_packed_sequence(sequence=pack_seq_rep, batch_first=True, total_length=max_seq_len)

        return self.dropout_last(output)
