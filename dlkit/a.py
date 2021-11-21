import torch
import torch.nn as nn

class Encoder(nn.Module):
    """docstring for Base"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.elinear1 = nn.Linear(3, 4)
        self.elinear2 = nn.Linear(4, 3)
        

class Base(nn.Module):
    """docstring for Base"""
    def __init__(self):
        super(Base, self).__init__()
        self.encoder = Encoder()

        self.decoder = Encoder()
        # self.conv = nn.Conv1d(4, 6, 3)
        self.apply(self.init_weight)
        self.linear = nn.Linear(4, 6)

    @staticmethod
    def init_weight(module):
        """TODO: Docstring for init_weight.
        :returns: TODO
        """
        if isinstance(module, nn.Linear):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 0)

        
a = Base()
print(list(a.parameters()))
