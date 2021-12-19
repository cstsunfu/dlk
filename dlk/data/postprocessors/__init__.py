"""postprocessors"""

import importlib
import os
from typing import Callable, Dict, Type
from dlk.utils.register import Register
from dlk.utils.config import BaseConfig
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

    def loss_name_map(self, stage):
        """TODO: Docstring for loss_name_map.
        :returns: TODO

        """
        map = {
            "valid": 'val',
            'train': 'train',
            "test": "test",
        }
        return map.get(stage, stage)

    def average_loss(self, list_batch_outputs):
        """TODO: Docstring for average_loss.

        :list_batch_outputs: TODO
        :returns: TODO

        """
        sum_loss = 0
        for batch_output in list_batch_outputs:
            sum_loss += batch_output.get('loss', 0)
        return sum_loss / len(list_batch_outputs)

    @abc.abstractmethod
    def do_predict(self, stage, list_batch_outputs, origin_data, rt_config):
        """TODO: Docstring for do_predict.
        :stage: TODO
        :list_batch_outputs: TODO
        :origin_data: TODO
        :rt_config: TODO
        :returns: TODO

        """
        raise NotImplementedError

    @abc.abstractmethod
    def do_calc_metrics(self, predicts, stage, list_batch_outputs, origin_data, rt_config):
        """TODO: Docstring for do_calc_metrics.
        :returns: TODO

        """
        raise NotImplementedError

    @abc.abstractmethod
    def do_save(self, predicts, stage, list_batch_outputs, origin_data, rt_config={}, save_condition=False):
        """TODO: Docstring for do_save.

        :predicts: TODO
        :rt_config: TODO
        :condition: when the save condition is True, do save
        :returns: TODO
        """
        raise NotImplementedError

    @property
    def without_ground_truth_stage(self):
        """TODO: Docstring for no_ground_truth_stage.
        :returns: TODO

        """
        return {'predict', 'online'}

    def process(self, stage, list_batch_outputs, origin_data, rt_config)->Dict:
        """TODO: Docstring for process.

        :arg1: TODO
        :returns: TODO

        """
        log_info = {}
        if stage not in self.without_ground_truth_stage:
            average_loss = self.average_loss(list_batch_outputs=list_batch_outputs)
            log_info[f'{self.loss_name_map(stage)}_loss'] = average_loss
        predicts = self.do_predict(stage, list_batch_outputs, origin_data, rt_config)
        if stage not in self.without_ground_truth_stage:
            log_info.update(self.do_calc_metrics(predicts, stage, list_batch_outputs, origin_data, rt_config))
        if stage == 'online':
            return predicts
        if stage == 'predict':
            self.do_save(predicts, stage, list_batch_outputs, origin_data, rt_config, save_condition=True)
        else:
            self.do_save(predicts, stage, list_batch_outputs, origin_data, rt_config, save_condition=False)
        return log_info

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
import_postprocessors(postprocessors_dir, "dlk.data.postprocessors")
