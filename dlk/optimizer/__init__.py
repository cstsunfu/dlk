# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
from typing import Any, Callable, Dict, Tuple, Type

import torch
import torch.optim as optim
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
    dataclass,
)

from dlk.utils.import_module import import_module_dir

logger = logging.getLogger(__name__)


@dataclass
class BaseOptimizerConfig(Base):
    """the base optimizer"""

    name = StrField(value="default", help="the name of default group parameters")
    optimizer_special_groups = DictField(
        value={},
        suggestions=[
            {
                "order": ["decoder", "bias"],
                "bias": {
                    "config": {"weight_decay": 0},
                    "pattern": ["bias", "LayerNorm.bias", "LayerNorm.weight"],
                },
                "decoder": {"config": {"lr": 1e-3}, "pattern": ["decoder"]},
            }
        ],
        help="""the special groups of optimizer, like
             "order": ['decoder', 'bias'], // the group order, if the para is in decoder & is in bias, set to decoder. The order name is set to the group name
             "bias": {
                 "config": { "weight_decay": 0 },
                 "pattern": ["bias",  "LayerNorm.bias", "LayerNorm.weight"]
             },
             "decoder": {
                 "config": { "lr": 1e-3 },
                 "pattern": ["decoder"]
             }""",
    )


class BaseOptimizer(object):
    def __init__(
        self,
        model: torch.nn.Module,
        config: BaseOptimizerConfig,
        optimizer: Type[optim.Optimizer],
    ):
        super(BaseOptimizer, self).__init__()
        self.config = config
        self.model = model
        self.optimizer = optimizer

    def get_optimizer(self) -> optim.Optimizer:
        """return the initialized optimizer

        Returns:
            Optimizer

        """
        return self.init_optimizer(self.optimizer, self.model, self.config)

    def init_optimizer(
        self,
        optimizer: Type[optim.Optimizer],
        model: torch.nn.Module,
        config: BaseOptimizerConfig,
    ) -> optim.Optimizer:
        """init the optimizer for paras in model, and the group is decided by config

        Args:
            optimizer: adamw, sgd, etc.
            model: pytorch model
            config: which decided the para group, lr, etc.

        Returns:
            the initialized optimizer

        """
        optimizer_special_groups = config.optimizer_special_groups
        params = []
        all_named_parameters = list(model.named_parameters())
        total_all_named_parameters = len(all_named_parameters)
        logger.info(f"All Named Params Num is {len(all_named_parameters)}")
        has_grouped_params = set()
        for special_group_name in optimizer_special_groups.get("order", []):
            group_config = optimizer_special_groups[special_group_name]["config"]
            group_patterns = optimizer_special_groups[special_group_name]["pattern"]
            # convert to regex
            combine_patterns = []
            for pattern in group_patterns:
                combine_patterns.append(f"({pattern})")
            cc_patterns = re.compile("|".join(combine_patterns))
            group_param = []
            for n, p in all_named_parameters:
                # logger.info(f"Param name {n}")
                if n in has_grouped_params:
                    continue
                if cc_patterns.search(n):  # use regex
                    has_grouped_params.add(n)
                    group_param.append(p)
            group_config["params"] = group_param
            group_config["name"] = special_group_name
            params.append(group_config)

        reserve_params = [
            p for n, p in all_named_parameters if not n in has_grouped_params
        ]
        params.append({"params": reserve_params, "name": config.name})
        logger.info(f"Param Group Nums {len(params)}")
        total_param = 0
        for group in params:
            total_param = total_param + len(group["params"])
        assert total_param == total_all_named_parameters
        opt_args = config._to_dict(only_para=True)
        opt_args.pop("name")
        opt_args.pop("optimizer_special_groups")

        return optimizer(params=params, **opt_args)

    def __call__(self):
        """the same as self.get_optimizer()"""
        return self.get_optimizer()


optimizer_dir = os.path.dirname(__file__)
import_module_dir(optimizer_dir, "dlk.optimizer")
