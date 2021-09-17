import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
from utils.config import Config
from . import encoder_register, encoder_config_register, EncoderInput, EncoderOutput 

@encoder_config_register("lstm")
class LSTMConfig(Config):
    """docstring for LSTMConfig"""
    def __init__(self, **args):
        super(LSTMConfig, self).__init__()
        self.num_layer = args.get('layer', 1)
        self.bidirect = args.get('bidirect', False)
        self.input_size = args.get('input_size', 128)
        self.output_size = args.get('output_size', 128)
        self.dropout = args.get('dropout', 0.1)
        

@encoder_register("lstm")
class LSTM(nn.Module):
    def __init__(self, config: LSTMConfig):
        super(LSTM, self).__init__()
        self.lstm = nn.ModuleList()
        hidden_size = config.output_size if not config.bidirect else config.output_size//2
        for i in range(config.num_layer):
            input_size = config.input_size if i == 0 else 2 * hidden_size
            self.lstm.append(nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=config.bidirect, dropout=self.dropout))

    def forward(self, inputs: EncoderInput)->EncoderOutput:
        """
        """
        # No padding necessary.
        mask = inputs.input_mask
        seq_rep = inputs.represent
        max_seq_len = seq_rep.size(1)
        seq_lens = max_seq_len - mask.sum(1)
        seq_rep = pack_padded_sequence(input=seq_rep, lengths=seq_lens, batch_first=True, enforce_sorted=False)
        for layer in self.lstm:
            seq_rep = layer(seq_rep)[0]
        seq_rep, _seq_lens = pad_packed_sequence(sequence=seq_rep, batch_first=True, total_length=max_seq_len)
        return EncoderOutput(represent=seq_rep)


if __name__ == "__main__":
    import os
    embedding = nn.Embedding(100, 6)
    input = torch.tensor([[1, 0, 0, 0, 0], [4, 2, 8, 0, 0], [2, 3, 0, 0, 0], [2, 3, 5, 7, 0]], dtype=torch.long)
    mask = input == 0
    input = embedding(input)
    print(input)
    lstm = LSTM(LSTMConfig(input_size=6, output_size=6))

    print(lstm(EncoderInput(represent=input, input_mask=mask)).represent)
    # print(input.size())

    # print(a.size(1))
