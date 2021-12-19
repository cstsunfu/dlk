import torch
import torch.nn as nn
from collections import namedtuple
from typing import Dict, List
import numpy as np
import abc
import sys
sys.path.append('../../')
from dlk.utils.base_module import BaseModule

class Module(BaseModule):
    """docstring for Module"""
    def __init__(self):
        super(Module, self).__init__()
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
# script.eval()
# print(script(d))
# # a = torch.randn(5, 3)
# nt = namedtuple("nt", ['inp', 'out'], defaults=None)
# out = nt._make(script(d))
# print(out.inp)
# print(out.out)
# from collections import namedtuple

# # 定义一个namedtuple类型User，并包含name，sex和age属性。
# User = namedtuple('User', ['name', 'sex', 'age'], defaults=[None]*3)

# # 创建一个User对象
# user = User(name='kongxx', sex='male')

# # # 也可以通过一个list来创建一个User对象，这里注意需要使用"_make"方法
# # user = User._make(['kongxx', 'male', 21])

# print(user)
# # User(name='user1', sex='male', age=21)

# def test(a):
    # """TODO: Docstring for test.

    # :a: TODO
    # :returns: TODO

    # """
    # print(a.inp)

# nt = namedtuple("nt", ['inp', 'out'], defaults=None)
# a = "111"
# d = nt(inp=a, out=a)
# test(d)
