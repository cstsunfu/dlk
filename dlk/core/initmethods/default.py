import torch.nn as nn
from . import initmethod_register, initmethod_config_register
from typing import Dict, List
from dlk.utils.logger import Logger
from dlk.utils.config import BaseConfig
import numpy as np
import torch

logger = Logger.get_logger()

# TODO:
@initmethod_config_register('default')
class RangeNormInitConfig(BaseConfig):
    """
        {
            "_name": "default",
            "config": {
            }
        }
    """
    def __init__(self, config):
        super(RangeNormInitConfig, self).__init__(config)
        self.post_check(config['config'])

@initmethod_register('default')
class RangeNormInit(object):
    """
    """

    def __init__(self, config: RangeNormInitConfig):
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
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
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
        # nn.init.xavier_uniform_(lstm.weight_hh_l0)
        # nn.init.xavier_uniform_(lstm.weight_hh_l0_reverse)
        # nn.init.xavier_uniform_(lstm.weight_ih_l0)
        # nn.init.xavier_uniform_(lstm.weight_ih_l0_reverse)
        # lstm.bias_hh_l0.data.fill_(0)
        # lstm.bias_hh_l0_reverse.data.fill_(0)
        # lstm.bias_ih_l0.data.fill_(0)
        # lstm.bias_ih_l0_reverse.data.fill_(0)
        # # Init forget gates to 1
        # for names in lstm._all_weights:
            # for name in filter(lambda n: 'bias' in n, names):
                # bias = getattr(lstm, name)
                # n = bias.size(0)
                # start, end = n // 4, n // 2
                # bias.data[start:end].fill_(1.)

        # Another init method
        for ind in range(0, lstm.num_layers):
            weight = eval('lstm.weight_ih_l' + str(ind))
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -bias, bias)
            weight = eval('lstm.weight_hh_l' + str(ind))
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -bias, bias)
        if lstm.bidirectional:
            for ind in range(0, lstm.num_layers):
                weight = eval('lstm.weight_ih_l' + str(ind) + '_reverse')
                bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
                nn.init.uniform_(weight, -bias, bias)
                weight = eval('lstm.weight_hh_l' + str(ind) + '_reverse')
                bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
                nn.init.uniform_(weight, -bias, bias)

        if lstm.bias:
            for ind in range(0, lstm.num_layers):
                weight = eval('lstm.bias_ih_l' + str(ind))
                weight.data.zero_()
                weight.data[lstm.hidden_size: 2 * lstm.hidden_size] = 1
                weight = eval('lstm.bias_hh_l' + str(ind))
                weight.data.zero_()
                weight.data[lstm.hidden_size: 2 * lstm.hidden_size] = 1
            if lstm.bidirectional:
                for ind in range(0, lstm.num_layers):
                    weight = eval('lstm.bias_ih_l' + str(ind) + '_reverse')
                    weight.data.zero_()
                    weight.data[lstm.hidden_size: 2 * lstm.hidden_size] = 1
                    weight = eval('lstm.bias_hh_l' + str(ind) + '_reverse')
                    weight.data.zero_()
                    weight.data[lstm.hidden_size: 2 * lstm.hidden_size] = 1
