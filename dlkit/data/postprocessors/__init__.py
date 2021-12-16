"""postprocessors"""

import importlib
import os
from typing import Callable, Dict, Type
from dlkit.utils.register import Register
from dlkit.utils.config import BaseConfig
import abc

class IPostProcessorConfig(BaseConfig):
    """docstring for PostProcessorConfigBase"""
    def __init__(self, config):
        super(IPostProcessorConfig, self).__init__(config)
        self.config = config.get('config', {})

    @property
    def input_map(self):
        """required the output of model process content name map
        :returns: TODO

        """
        return self.config.get("input_map", {})

    @property
    def origin_input_map(self):
        """required the origin data(before pass to datamodule) column name map
        :returns: TODO

        """
        return self.config.get("origin_input_map", {})


class IPostProcessor(metaclass=abc.ABCMeta):
    """docstring for IPostProcessor"""

    def loss_name_map(self, stage_name):
        """TODO: Docstring for loss_name_map.

        :stage: TODO
        :returns: TODO

        """
        loss_name_map = {
            "valid": "val_loss",
            "train": "train_loss",
            "test": "test_loss"
        }
        return loss_name_map.get(stage_name, stage_name+'_loss')

    def average_loss(self, list_batch_outputs):
        """TODO: Docstring for average_loss.

        :list_batch_outputs: TODO
        :returns: TODO

        """
        sum_loss = 0
        for batch_output in list_batch_outputs:
            sum_loss += batch_output.get('loss', 0)
        return sum_loss / len(list_batch_outputs)


    @property
    def without_ground_truth_stage(self):
        """TODO: Docstring for no_ground_truth_stage.
        :returns: TODO

        """
        return {'predict', 'online'}

    @abc.abstractmethod
    def process(self, stage, list_batch_outputs, origin_data, rt_config)->Dict:
        """TODO: Docstring for process.

        :arg1: TODO
        :returns: TODO

        """
        raise NotImplementedError

    def __call__(self, stage, list_batch_outputs, origin_data, rt_config):
        """TODO: Docstring for __call.
        :returns: TODO

        """
        return self.process(stage, list_batch_outputs, origin_data, rt_config)


postprocessor_config_register = Register('PostProcessor config register')
postprocessor_register = Register("PostProcessor register")


def import_postprocessors(postprocessors_dir, namespace):
    for file in os.listdir(postprocessors_dir):
        path = os.path.join(postprocessors_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            # and not (file.endswith("subpostprocessors") and os.path.isdir(path))
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            postprocessor_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + postprocessor_name)


# automatically import any Python files in the models directory
postprocessors_dir = os.path.dirname(__file__)
import_postprocessors(postprocessors_dir, "dlkit.data.postprocessors")
