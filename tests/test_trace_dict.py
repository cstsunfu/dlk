import torch
import torch.nn as nn
from collections import namedtuple
from typing import Dict, List
import numpy as np
import abc
import sys
from dlk.core.base_module import BaseModule

class Module(BaseModule):
    """docstring for Module"""
    def __init__(self):
        super(BaseModule, self).__init__()
        self.linear = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 3)
        self.config = {}
        self.config['output_map'] = {"inp":"map_inp"}
        # self._name_map = {"inp":"map_inp"}
        self.provide = {''}
        self.required = {''}

    def training_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """training
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        raise NotImplementedError

    def validation_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """valid
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        raise NotImplementedError

    def provide_keys(self)->List[str]:
        """TODO: Docstring for provide_keys.
        :returns: TODO
        """
        pass
        # raise NotImplementedError

    def check_keys_are_provided(self, provide: List[str])->bool:
        """TODO: Docstring for check_key_are_provided.
        :returns: TODO
        """
        pass

    def predict_step(self, args: Dict[str, torch.Tensor]):
        """TODO: Docstring for forward.

        :**args: TODO
        :returns: TODO

        """
        inp = args.pop('inp')
        out = args.pop('out')
        if self.training:
            print('Train')
        else:
            print("eval")

        x = self.linear(inp)
        x = self.linear2(x)
        # print(x)
        # nt = namedtuple("nt", ['inp', 'out'], defaults=[None]*2)
        result = {"inp": inp, "out": out, "x": x}
        # result = {self.name_map.get(key, key): value for key, value in result.items()}
        return self.output_rename(result)
        # return nt(out=out)
if __name__ == "__main__":
    
    a = torch.randn(1, 3)
    d = {"inp":a, "out":a}

    model = Module()
    # model(d)
    # print(d)
    # script = torch.jit.trace(model, d, strict=False)
    model.eval()
    # print(model(d))
    script = torch.jit.script(model)
    # script.train()

    print(script(d))
