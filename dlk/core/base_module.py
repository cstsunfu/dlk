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
        self._logits_gather_config = module_config_register.get("logits_gather")(config['config'].pop("logits_gather_config", {}))


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

    def get_real_name(self, name, name_map):
        """TODO: Docstring for get_real_name.
        :returns: TODO

        """
        if name in name_map:
            return name_map[name]
        else:
            return name

    def get_input_name(self, name):
        """TODO: Docstring for get_input_name.
        :name: TODO
        :returns: TODO
        """
        return self.get_real_name(name, self.config._input_map)

    def get_output_name(self, name):
        """TODO: Docstring for get_input_name.
        :name: TODO
        :returns: TODO
        """
        return self.get_real_name(name, self.config._output_map)

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

class BaseModel(nn.Module, ModuleOutputRenameMixin, IModuleIO, IModuleStep):
    """docstring for BaseModule"""

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """predict forward
        """
        raise NotImplementedError

class BaseModule(nn.Module, ModuleOutputRenameMixin, IModuleIO, IModuleStep):
    """docstring for BaseModule"""

    def __init__(self, config: BaseModuleConfig):
        super(BaseModule, self).__init__()
        self._logits_gather = module_register.get("logits_gather")(config._logits_gather_config)
        self._provided_keys = set()
        self._required_keys = set()
        self._provide_keys = set()
        self.config = config # for better complete, you can rewrite this in child module

    def provide_keys(self)->Set:
        """TODO: should provide_keys in model?
        """
        return self.set_rename(self._provide_keys, self.config._output_map).union(self._provided_keys)

    def check_keys_are_provided(self, provide: Set[str])->None:
        """
        """
        self._provided_keys = provide
        provide = self.set_rename(provide, self.config._input_map)
        for required_key in self._required_keys:
            if required_key not in provide:
                raise PermissionError(f"The {self.__class__.__name__} Module required '{required_key}' as input.")

    def init_weight(self, method: Callable):
        """init  Module weight by `method`
        :method: init method
        :returns: None
        """
        for module in self.children():
            module.apply(method)

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
