import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
from dlkit.utils.config import Config
from dlkit.modules.decoders import decoder_register, decoder_config_register, DecoderInput, DecoderOutput 

@decoder_config_register("linear")
class LinearConfig(Config):
    """docstring for LinearConfig"""
    def __init__(self, **args):
        super(LinearConfig, self).__init__()
        self.num_layer = args.get('layer', 1)
        self.input_size = args.get('input_size', 128)
        self.output_size = args.get('output_size', 128)
        

@decoder_register("linear")
class Linear(nn.Module):
    def __init__(self, config: LinearConfig):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=config.input_size, out_features=config.output_size)

    def forward(self, inputs: DecoderInput)->DecoderOutput:
        """
        """
        # No padding necessary.
        seq_rep = inputs.represent
        seq_rep = self.linear(seq_rep)
        return DecoderOutput(represent=seq_rep)


if __name__ == "__main__":
    import os
    embedding = nn.Embedding(100, 6)
    input = torch.tensor([[1, 0, 0, 0, 0], [4, 2, 8, 0, 0], [2, 3, 0, 0, 0], [2, 3, 5, 7, 0]], dtype=torch.long)
    mask = input == 0
    input = embedding(input)
    print(input)
    linear = Linear(LinearConfig(input_size=6, output_size=6))

    print(linear(DecoderInput(represent=input, input_mask=mask)).represent)
    # print(input.size())

    # print(a.size(1))
