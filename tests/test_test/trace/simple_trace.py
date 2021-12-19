import torch
import torch.nn as nn

class Module(nn.Module):
    """docstring for Module"""
    def __init__(self):
        super(Module, self).__init__()
        self.linear = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 3)

    def forward(self, inp, out):
        """TODO: Docstring for forward.

        :**args: TODO
        :returns: TODO

        """
        # inp = args.pop('inp')
        x = self.linear(inp)
        x = self.linear2(x)
        return x, out

a = torch.randn(2, 1, 3)

model = Module()

script = torch.jit.trace(model, (a, a))
a = torch.randn(5, 3)
# print(script(a, a))
