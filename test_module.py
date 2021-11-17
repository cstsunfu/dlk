import torch
import torch.nn as nn

def make_linear(input_dim, output_dim):
    """TODO: Docstring for make_linear.

    :input_dim: TODO
    :output_dim: TODO
    :returns: TODO

    """
    linear = nn.Linear(input_dim, output_dim)
    return linear


class Module(nn.Module):
    """docstring for Module"""
    def __init__(self):
        super(Module, self).__init__()
        self.layer1 = make_linear(3, 3)
        self.layer2 = nn.Linear(3, 8)
        
    def forward(self):
        """TODO: Docstring for forward.

        :f: TODO
        :returns: TODO

        """
        return


model = Module()
for p in model.named_parameters():
    print(p[0])
