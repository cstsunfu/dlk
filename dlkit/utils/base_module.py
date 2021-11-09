import torch
import torch.nn as nn
from typing import Dict, List
import abc


class ModuleOutputRenameMixin:
    """Just rename the output key name by config to adapt the input field of downstream module."""
    def output_rename(self, result: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        return {self.config['output_map'].get(key, key): value for key, value in result.items()}
        

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

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """predict forward
        """
        return self.predict_step(inputs)
