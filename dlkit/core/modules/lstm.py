import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict
from . import module_register, module_config_register

@module_config_register("lstm")
class LSTMConfig(object):
    """docstring for LSTMConfig
    {
        config: {
            bidirectional: true,
            hidden_size: 200, //the output is 2*hidden_size if use
            input_size: 200,
            proj_size: 200,
            num_layers: 1,
            dropout: 0.1, // dropout between layers
            dropout_last: true, //dropout the last layer output or not
        },
        _name: "lstm",
    }
    """

    def __init__(self, config: Dict):
        super(LSTMConfig, self).__init__()
        config = config.get('config', {})
        self.num_layers = config.get('num_layers', 1)
        self.bidirectional= config.get('bidirectional', False)
        self.input_size = config.get('input_size', 128)
        self.hidden_size = config.get('hidden_size', 128)
        self.proj_size = config.get('proj_size', 128)
        self.dropout = config.get('dropout', 0.1)
        self.dropout_last = config.get('dropout_last', False)
        

@module_register("lstm")
class LSTM(nn.Module):
    def __init__(self, config: LSTMConfig):
        super(LSTM, self).__init__()
        # the output_size is proj_size // 2 when bidirectinal is true
        if config.bidirectional:
            assert config.proj_size % 2 == 0
            config.proj_size = config.proj_size // 2

        self.lstm = nn.LSTM(config.input_size, config.hidden_size, num_layers=config.num_layers, bidirectional=config.bidirectional, dropout=config.dropout, proj_size=config.proj_size)
        self.dropout_last = nn.Dropout(p=config.dropout if config.dropout_last else 0)

    def forward(self, input: torch.Tensor, mask: torch.Tensor)->torch.Tensor:
        """
        """
        # No padding necessary.
        max_seq_len = input.size(1)
        seq_lens = max_seq_len - mask.sum(1)
        pack_seq_rep = pack_padded_sequence(input=input, lengths=seq_lens, batch_first=True, enforce_sorted=False)
        pack_seq_rep = self.lstm(pack_seq_rep)[0]
        output, _ = pad_packed_sequence(sequence=pack_seq_rep, batch_first=True, total_length=max_seq_len)
        return self.dropout_last(output)