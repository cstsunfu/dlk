import torch.nn as nn
from . import callback_register, callback_config_register
from typing import Dict, List
import os
from pytorch_lightning.callbacks import ProgressBarBase


@callback_config_register('progressbar')
class ProgressBarCallbackConfig(object):
    """
    {
        // default progressbar configure
        "_name": progressbar,
        "config": {
            "disable_v_num": true,
        }
    }
    """
    def __init__(self, config):
        super(ProgressBarCallbackConfig, self).__init__()
        config = config['config']
        self.disable_v_num = config['disable_v_num']



@callback_register('progressbar')
class ProgressBarCallback(object):
    """
    """

    def __init__(self, config: ProgressBarCallbackConfig):
        super().__init__()
        self.config = config

    def __call__(self, rt_config: Dict):
        """TODO: Docstring for __call__.

        :rt_config: Dict: TODO
        :returns: TODO

        """
        raise PermissionError(f"Don't use Progressbar callback, it's not implemented yet.")
        return ProgressBarBase()
