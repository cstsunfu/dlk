import torch
import torch.nn as nn
from typing import Dict, List, Callable, Any, Set
import abc


class ModuleOutputRenameMixin:
    """Just rename the output key name by config to adapt the input field of downstream module."""
    def dict_rename(self, input: Dict, output_map: Dict[str, str])->Dict:
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

    def set_rename(self, input: Set, output_map: Dict[str, str])->Set:
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
    """docstring for IModuleIO"""

    @abc.abstractmethod
    def provide_keys(self)->List[str]:
        """TODO: Docstring for provide_keys.
        :returns: TODO
        """
        pass

    @abc.abstractmethod
    def check_keys_are_provided(self, provide: List[str])->bool:
        """TODO: Docstring for check_keys_are_provided.
        :returns: TODO
        """
        pass

    def check_module_chain(self, module_list: List["BaseModule"]):
        """check the interfaces of the list of modules are alignd or not.

        :List[nn.Module]: all modules in this chain
        :returns: None, if check not passed, raise a ValueError
        """
        assert len(module_list) > 1
        result = True
        for i in range(len(module_list)-1):
            result = result and module_list[i+1].check_keys_are_provided(module_list[i].provide_keys())
            if not result:
                raise ValueError(f'The module "{module_list[i+1]._name}" is required "{", ".join(module_list[i+1].provide_keys())}", \
                    but the module "{module_list[i]._name}" provide "{", ".join(module_list[i].provide_keys())}"! ')


class IModuleStep(metaclass=abc.ABCMeta):
    """docstring for ModuleStepMixin"""


    @abc.abstractmethod
    def predict_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """predict
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        raise NotImplementedError
        
    @abc.abstractmethod
    def training_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """training
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        raise NotImplementedError

    @abc.abstractmethod
    def validation_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """valid
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        raise NotImplementedError

    def test_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """valid
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        return self.validation_step(inputs)


class BaseModule(nn.Module, ModuleOutputRenameMixin, IModuleIO, IModuleStep):
    """docstring for BaseModule"""

    def init_weight(self, method: Callable):
        """init  Module weight by `method`
        :method: init method
        :returns: None
        """
        self.apply(method)

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """predict forward
        """
        raise NotImplementedError


class SimpleModule(BaseModule):
    """docstring for SimpleModule, SimpleModule, all train/predict/test/validation step call the forward"""

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """predict forward
        """
        raise NotImplementedError

    def predict_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """predict
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        return self(inputs)
        
    def training_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """training
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        return self(inputs)

    def validation_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """valid
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        return self(inputs)

    def test_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """valid
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        return self(inputs)