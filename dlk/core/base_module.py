# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Any, Set
from dlk.core.modules import module_register, module_config_register
from dlk.utils.logger import Logger
from dlk.utils.config import BaseConfig
import abc

logger = Logger.get_logger()


class BaseModuleConfig(BaseConfig):
    """docstring for BaseLayerConfig"""
    def __init__(self, config: Dict):
        super(BaseModuleConfig, self).__init__(config)
        self._output_map = config['config'].pop("output_map", {})
        self._input_map = config['config'].pop('input_map', {})
        self._logits_gather_config = module_config_register.get("logits_gather")(config.pop("module@logits_gather", {}))


class ModuleOutputRenameMixin:
    """Just rename the output key name by config to adapt the input field of downstream module."""
    def dict_rename(self, input: Dict, output_map: Dict[str, str])->Dict:
        """rename the key of input(dict) by output_map(name map)

        Args:
            input: will rename input
            output_map:  name map

        Returns: 
            renamed input
        """
        if isinstance(input, dict):
            output = {}
            for key, value in input.items():
                if key in output_map:
                    output[output_map[key]] = value
                else:
                    output[key] = value
            return output
        else:
            raise PermissionError("Not Defined")

    def get_real_name(self, name: str, name_map: Dict[str, str])->str:
        """use the name_map to map the input name to real name

        Args:
            name: input_name
            name_map: name map
        Returns: 
            real_name

        """
        if name in name_map:
            return name_map[name]
        else:
            return name

    def get_input_name(self, name: str)->str:
        """use config._input_map map the name to real name

        Args:
            name: input_name

        Returns: 
            real_name

        """
        return self.get_real_name(name, self.config._input_map)

    def get_output_name(self, name:str)->str:
        """use config._output_map map the name to real name

        Args:
            name: output_name

        Returns: 
            real_name

        """
        return self.get_real_name(name, self.config._output_map)

    def set_rename(self, input: Set, output_map: Dict[str, str])->Set:
        """rename all the name in input by output_map

        Args:
            input: a set of names
            output_map: name map

        Returns: 
            renamed input

        """
        if isinstance(input, set):
            output = set()
            for key in input:
                if key in output_map:
                    output.add(output_map[key])
                else:
                    output.add(key)
            return output
        else:
            raise PermissionError("Not Defined")


class IModuleIO(metaclass=abc.ABCMeta):
    """interface for check the modules input and output"""

    @abc.abstractmethod
    def provide_keys(self)->List[str]:
        """return all keys of the dict of the module returned

        Returns: 
            all keys
        """
        pass

    @abc.abstractmethod
    def check_keys_are_provided(self, provide: List[str])->bool:
        """check this module required key are provided

        Returns: 
            pass or not

        """
        pass

    def check_module_chain(self, module_list: List['BaseModule'])->bool:
        """check the interfaces of the list of modules are alignd or not.

        Args:
            module_list: a series modules

        Returns: 
            pass or not

        Raises:
            ValueError: the check is not passed
        """
        assert len(module_list) > 1
        result = True
        for i in range(len(module_list)-1):
            result = result and module_list[i+1].check_keys_are_provided(module_list[i].provide_keys())
            if not result:
                raise ValueError(f'The module "{module_list[i+1]._name}" is required "{", ".join(module_list[i+1].provide_keys())}", \
                    but the module "{module_list[i]._name}" provide "{", ".join(module_list[i].provide_keys())}"! ')
        return True


class IModuleStep(metaclass=abc.ABCMeta):
    """docstring for ModuleStepMixin"""


    @abc.abstractmethod
    def predict_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do predict for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: 
            the predicts outputs

        """
        raise NotImplementedError

    @abc.abstractmethod
    def training_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do training for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        raise NotImplementedError

    @abc.abstractmethod
    def validation_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do validataion for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        raise NotImplementedError

    def test_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do test for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        return self.validation_step(inputs)

class BaseModel(nn.Module, ModuleOutputRenameMixin, IModuleIO, IModuleStep):
    """All pytorch models should inheritance this class
    """

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """all models should apply this method

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        raise NotImplementedError

class BaseModule(nn.Module, ModuleOutputRenameMixin, IModuleIO, IModuleStep):
    """All pytorch modules should inheritance this class
    """

    def __init__(self, config: BaseModuleConfig):
        super(BaseModule, self).__init__()
        self._logits_gather = module_register.get("logits_gather")(config._logits_gather_config)
        self._provided_keys = set()
        self._required_keys = set()
        self._provide_keys = set()
        self.config = config # for better complete, you can rewrite this in child module

    def provide_keys(self)->Set:
        """return all keys of the dict of the module returned

        Returns: 
            all keys
        """
        return self.set_rename(self._provide_keys, self.config._output_map).union(self._provided_keys)

    def check_keys_are_provided(self, provide: Set[str])->None:
        """check this module required key are provided

        Returns: 
            pass or not
        """
        self._provided_keys = provide
        provide = self.set_rename(provide, self.config._input_map)
        for required_key in self._required_keys:
            if required_key not in provide:
                raise PermissionError(f"The {self.__class__.__name__} Module required '{required_key}' as input.")

    def init_weight(self, method):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns: 
            None

        """
        for module in self.children():
            module.apply(method)

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """all module should apply this method

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        raise NotImplementedError


class SimpleModule(BaseModule):
    """docstring for SimpleModule, SimpleModule, all train/predict/test/validation step call the forward"""

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """in simple module, all step fit to this method

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        raise NotImplementedError

    def predict_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do predict for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        return self(inputs)

    def training_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do train for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        return self(inputs)

    def validation_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do validation for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        return self(inputs)

    def test_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do test for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        return self(inputs)
