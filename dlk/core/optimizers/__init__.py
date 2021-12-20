"""optimizers"""
import importlib
import os
from typing import Callable, Dict, Tuple, Any
from dlk.utils.register import Register
import torch.optim as optim
from dlk.utils.logger import Logger
import re


logger = Logger.get_logger()
optimizer_config_register = Register("Optimizer config register.")
optimizer_register = Register("Optimizer register.")


class BaseOptimizer(object):

    def get_optimizer(self):
        """TODO: Docstring for get_optimizer.
        :returns: TODO

        """
        raise NotADirectoryError

    def init_optimizer(self, optimizer, model, config):
        """TODO: Docstring for get_optimizer.
        :returns: TODO
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
        """TODO: Docstring for __call__.
        :returns: TODO

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
