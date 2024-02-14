# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List

import numpy as np
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

from dlk.utils.register import register

logger = logging.getLogger(__name__)


@cregister("initmethod", "default")
class DefaultInitConfig:
    """
    The default init method for the modules
    """

    pass


@register("initmethod", "default")
class DefaultInit(object):
    """default method for init the modules"""

    def __init__(self, config: DefaultInitConfig):
        super().__init__()

    def __call__(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # use the default kaiming init method
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            # use the default init
            # weight are initialized to 1, bias to 0
            module.reset_parameters()
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_uniform_(module.weight)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight)
        elif isinstance(module, nn.Conv3d):
            torch.nn.init.kaiming_uniform_(module.weight)
        elif isinstance(module, nn.LSTM):
            self.init_lstm(module)
        elif isinstance(module, nn.ModuleList) or isinstance(module, nn.Sequential):
            for sub_module in module:
                self(sub_module)
        elif isinstance(module, nn.ModuleDict):
            for sub_module in module:
                self(module[sub_module])
        else:
            logger.info(f"{module} is not initialization.")

    def init_lstm(self, lstm):
        """
        Initialize lstm
        """
        for ind in range(0, lstm.num_layers):
            weight = eval("lstm.weight_ih_l" + str(ind))
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -bias, bias)
            weight = eval("lstm.weight_hh_l" + str(ind))
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -bias, bias)
        if lstm.bidirectional:
            for ind in range(0, lstm.num_layers):
                weight = eval("lstm.weight_ih_l" + str(ind) + "_reverse")
                bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
                nn.init.uniform_(weight, -bias, bias)
                weight = eval("lstm.weight_hh_l" + str(ind) + "_reverse")
                bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
                nn.init.uniform_(weight, -bias, bias)

        if lstm.bias:
            for ind in range(0, lstm.num_layers):
                weight = eval("lstm.bias_ih_l" + str(ind))
                weight.data.zero_()
                weight.data[lstm.hidden_size : 2 * lstm.hidden_size] = 1
                weight = eval("lstm.bias_hh_l" + str(ind))
                weight.data.zero_()
                weight.data[lstm.hidden_size : 2 * lstm.hidden_size] = 1
            if lstm.bidirectional:
                for ind in range(0, lstm.num_layers):
                    weight = eval("lstm.bias_ih_l" + str(ind) + "_reverse")
                    weight.data.zero_()
                    weight.data[lstm.hidden_size : 2 * lstm.hidden_size] = 1
                    weight = eval("lstm.bias_hh_l" + str(ind) + "_reverse")
                    weight.data.zero_()
                    weight.data[lstm.hidden_size : 2 * lstm.hidden_size] = 1
