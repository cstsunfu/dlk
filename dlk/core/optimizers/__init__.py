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

"""optimizers"""
import importlib
import os
from typing import Callable, Dict, Tuple, Any, Type
from dlk.utils.register import Register
import torch.optim as optim
from dlk.utils.logger import Logger
import torch
import re
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules
logger = Logger.get_logger()


@define
class BaseOptimizerConfig(BaseConfig):
    name = NameField(value="*@*", file=__file__, help="the base optimizer")
    @define
    class Config:
        name = StrField(value="default", help="the name of default group parameters")
        optimizer_special_groups = DictField(value={}, checker=suggestions([
            {
                "order": ['decoder', 'bias'], 
                "bias": {
                    "config": { "weight_decay": 0 },
                    "pattern": ["bias",  "LayerNorm.bias", "LayerNorm.weight"]
                },
                "decoder": {
                    "config": { "lr": 1e-3 },
                    "pattern": ["decoder"]
                }
                }]), help="""the special groups of optimizer, like
                                             "order": ['decoder', 'bias'], // the group order, if the para is in decoder & is in bias, set to decoder. The order name is set to the group name
                                             "bias": {
                                                 "config": { "weight_decay": 0 },
                                                 "pattern": ["bias",  "LayerNorm.bias", "LayerNorm.weight"]
                                                 },
                                             "decoder": {
                                                 "config": { "lr": 1e-3 },
                                                 "pattern": ["decoder"]
                                                 }""")
    config = NestField(value=Config, converter=nest_converter)


class BaseOptimizer(object):
    def __init__(self, model: torch.nn.Module, config: BaseOptimizerConfig, optimizer: Type[optim.Optimizer]):
        super(BaseOptimizer, self).__init__()
        self.config = config.to_dict()['config']
        self.model = model
        self.optimizer = optimizer

    def get_optimizer(self)->optim.Optimizer:
        """return the initialized optimizer

        Returns: 
            Optimizer

        """
        return self.init_optimizer(self.optimizer, self.model, self.config)

    def init_optimizer(self, optimizer: Type[optim.Optimizer], model: torch.nn.Module, config: Dict)->optim.Optimizer:
        """init the optimizer for paras in model, and the group is decided by config

        Args:
            optimizer: adamw, sgd, etc.
            model: pytorch model
            config: which decided the para group, lr, etc.

        Returns: 
            the initialized optimizer

        """
        optimizer_special_groups = config.pop('optimizer_special_groups', {})
        params = []
        all_named_parameters = list(model.named_parameters())
        total_all_named_parameters = len(all_named_parameters)
        logger.info(f"All Named Params Num is {len(all_named_parameters)}")
        has_grouped_params = set()
        for special_group_name in optimizer_special_groups.get('order', []):

            group_config = optimizer_special_groups[special_group_name]['config']
            group_patterns = optimizer_special_groups[special_group_name]['pattern']
            # convert to regex
            combine_patterns = []
            for pattern in group_patterns:
                combine_patterns.append(f"({pattern})")
            cc_patterns = re.compile("|".join(combine_patterns))
            group_param = []
            for n, p  in all_named_parameters:
                # logger.info(f"Param name {n}")
                if n in has_grouped_params:
                    continue
                if cc_patterns.search(n): # use regex
                    has_grouped_params.add(n)
                    group_param.append(p)
            group_config['params'] = group_param
            group_config['name'] = special_group_name
            params.append(group_config)

        reserve_params = [p for n, p in all_named_parameters if not n in has_grouped_params]
        params.append({"params": reserve_params, "name": config.pop('name')})
        logger.info(f"Param Group Nums {len(params)}")
        total_param = 0
        for group in params:
            total_param = total_param + len(group['params'])
        assert total_param == total_all_named_parameters

        return optimizer(params=params, **config)

    def __call__(self):
        """the same as self.get_optimizer()
        """
        return self.get_optimizer()


def import_optimizers(optimizers_dir, namespace):
    for file in os.listdir(optimizers_dir):
        path = os.path.join(optimizers_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            optimizer_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + optimizer_name)


# automatically import any Python files in the optimizers directory
optimizers_dir = os.path.dirname(__file__)
import_optimizers(optimizers_dir, "dlk.core.optimizers")
