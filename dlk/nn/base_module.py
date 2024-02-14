# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging
from typing import Any, Callable, Dict, List, Set

import torch
import torch.nn as nn
from intc import (
    MISSING,
    AnyField,
    Base,
    BoolField,
    DictField,
    FloatField,
    IntField,
    ListField,
    NestField,
    StrField,
    SubModule,
    cregister,
)

from dlk.utils.register import register, register_module_name

logger = logging.getLogger(__name__)


class ModuleOutputRenameMixin:
    """Just rename the output key name by config to adapt the input field of downstream module."""

    def dict_rename(self, input: Dict, output_map: Dict[str, str]) -> Dict:
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

    def get_real_name(self, name: str, name_map: Dict[str, str]) -> str:
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

    def get_input_name(self, name: str) -> str:
        """use config._input_map map the name to real name

        Args:
            name: input_name

        Returns:
            real_name

        """
        return self.get_real_name(name, self.__input_map)

    def get_output_name(self, name: str) -> str:
        """use config._output_map map the name to real name

        Args:
            name: output_name

        Returns:
            real_name

        """
        return self.get_real_name(name, self.__output_map)

    def set_rename(self, input: Set, output_map: Dict[str, str]) -> Set:
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


class GatherOutputMixin(object):
    """gather all the small batches output to a big batch"""

    @staticmethod
    def proc_dist_outputs(dist_outputs: List[Dict]) -> List[Dict]:
        """gather all distributed outputs to outputs which is like in a single worker.

        Args:
            dist_outputs: the inputs of pytorch_lightning train/test/.._epoch_end when using ddp
        Returns:
            the inputs of pytorch_lightning train/test/.._epoch_end when only run on one worker.

        """
        outputs = []
        for dist_output in dist_outputs:
            one_output = {}
            for key in dist_output:
                try:
                    one_output[key] = torch.cat(
                        torch.swapaxes(dist_output[key], 0, 1).unbind(), dim=0
                    )
                except:
                    raise KeyError(f"{key}: {dist_output[key]}")
            outputs.append(one_output)
        return outputs

    def gather_outputs(self, outputs: List[Dict]):
        """gather the dist outputs

        Args:
            outputs: one node outputs

        Returns:
            all outputs

        """
        if self.trainer.world_size > 1:
            dist_outputs = self.all_gather(
                outputs
            )  # WARN: must padding all the dim to same except batch_size, otherwise will be trunc by the default all_gather function. Or check pytorch_lightning/utilities/distributed.gather_all_tensors
            if self.local_rank in [0, -1]:
                outputs = self.proc_dist_outputs(dist_outputs)
        return outputs

    def concat_list_of_dict_outputs(self, outputs: List[Dict]) -> Dict:
        """only support all the outputs has the same dim, now is deprecated.

        Args:
            outputs: multi node returned output (list of dict)

        Returns:
            Concat all list by name

        """
        key_all_batch_map = {}
        for batch in outputs:
            for key in batch:
                if key not in key_all_batch_map:
                    key_all_batch_map[key] = []
                key_all_batch_map[key].append(batch[key])
        key_all_ins_map = {}
        for key in key_all_batch_map:
            key_all_ins_map[key] = torch.cat(key_all_batch_map[key], dim=0)

        return key_all_ins_map


class IModuleIO(metaclass=abc.ABCMeta):
    """Currently Deprecated:
    interface for check the modules input and output"""

    @abc.abstractmethod
    def provide_keys(self) -> List[str]:
        """return all keys of the dict of the module returned

        Returns:
            all keys
        """
        pass

    @abc.abstractmethod
    def check_keys_are_provided(self, provide: List[str]) -> bool:
        """check this module required key are provided

        Returns:
            pass or not

        """
        pass

    def check_module_chain(self, module_list: List["BaseModule"]) -> bool:
        """check the interfaces of the list of modules are aligned or not.

        Args:
            module_list: a series modules

        Returns:
            pass or not

        Raises:
            ValueError: the check is not passed
        """
        assert len(module_list) > 1
        result = True
        for i in range(len(module_list) - 1):
            result = result and module_list[i + 1].check_keys_are_provided(
                module_list[i].provide_keys()
            )
            if not result:
                raise ValueError(
                    f'The module "{module_list[i+1]._name}" is required "{", ".join(module_list[i+1].provide_keys())}", \
                    but the module "{module_list[i]._name}" provide "{", ".join(module_list[i].provide_keys())}"! '
                )
        return True


class IModuleStep(metaclass=abc.ABCMeta):
    """docstring for ModuleStepMixin"""

    @abc.abstractmethod
    def predict_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do predict for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            the predicts outputs

        """
        raise NotImplementedError

    @abc.abstractmethod
    def training_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do training for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        raise NotImplementedError

    @abc.abstractmethod
    def validation_step(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """do validation for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        raise NotImplementedError

    def test_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do test for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        return self.validation_step(inputs)


class BaseModel(nn.Module, IModuleStep):
    """All pytorch models should inheritance this class"""

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """all models should apply this method

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        raise NotImplementedError


class BaseModule(nn.Module, IModuleStep):
    """All pytorch modules should inheritance this class"""

    def __init__(self, config: Base):
        super(BaseModule, self).__init__()
        dict_config = config._to_dict()
        self.__logits_gather = None
        for key, config in config._get_named_modules("module").items():
            if register_module_name(config._module_name) == "logits_gather":
                self.__logits_gather = register("module", "logits_gather")(config)

    def reorder_encoder_out(self, encoder_outs: Dict[str, torch.Tensor], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        raise NotImplementedError

    def init_weight(self, method):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        for module in self.children():
            module.apply(method)

    def forward_logits_gather(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """all module should apply this method

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        inputs = self.forward(inputs)
        if self.__logits_gather is not None:
            inputs = self.__logits_gather(inputs)
        return inputs

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """in simple module, all step fit to this method

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        raise NotImplementedError


class SimpleModule(BaseModule):
    """SSimpleModule, all train/predict/test/validation step call the forward"""

    def predict_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do predict for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        return self(inputs)

    def training_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do train for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        return self(inputs)

    def validation_step(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """do validation for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        return self(inputs)

    def test_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do test for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        return self(inputs)


class BaseIdentityModule(SimpleModule):
    """docstring for IdentityModule"""

    def __init__(self, config=Base()):
        super(BaseIdentityModule, self).__init__(config)

    def init_weight(self, method):
        pass

    def forward(self, inputs):
        return inputs

    def forward_wrap(self, inputs):
        return inputs
