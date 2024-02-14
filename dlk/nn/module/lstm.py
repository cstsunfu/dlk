# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from dlk.utils.register import register

from . import Module


@cregister("module", "lstm")
class LSTMConfig:
    """the lstm config"""

    bidirectional = BoolField(value=True, help="whether to use bidirectional lstm")
    input_size = IntField(value=MISSING, minimum=0, help="the input size")
    output_size = IntField(value=MISSING, minimum=0, help="the output size")
    num_layers = IntField(value=1, minimum=1, help="the number of layers")
    dropout = FloatField(value=0.0, minimum=0.0, maximum=1.0, help="the dropout rate")
    dropout_last = BoolField(
        value=True, help="whether to dropout the last layer output"
    )


@register("module", "lstm")
class LSTM(Module):
    "A wrap for nn.LSTM"

    def __init__(self, config: LSTMConfig):
        super(LSTM, self).__init__()
        self.config = config

        if self.config.num_layers <= 1:
            inlstm_dropout = 0
        else:
            inlstm_dropout = self.config.dropout

        hidden_size = self.config.output_size
        if self.config.bidirectional:
            assert self.config.output_size % 2 == 0
            hidden_size = self.config.output_size // 2

        self.lstm = nn.LSTM(
            input_size=self.config.input_size,
            hidden_size=hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=self.config.bidirectional,
            dropout=inlstm_dropout,
        )
        self.dropout_last = nn.Dropout(
            p=float(self.config.dropout) if self.config.dropout_last else 0
        )

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns:
            lstm output the shape is the same as input

        """
        max_seq_len = input.size(1)
        seq_lens = mask.sum(1).cpu()
        pack_seq_rep = pack_padded_sequence(
            input=input, lengths=seq_lens, batch_first=True, enforce_sorted=False
        )
        pack_seq_rep = self.lstm(pack_seq_rep)[0]
        output, _ = pad_packed_sequence(
            sequence=pack_seq_rep, batch_first=True, total_length=max_seq_len
        )

        return self.dropout_last(output)
