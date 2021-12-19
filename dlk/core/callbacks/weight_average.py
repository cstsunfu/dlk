import torch.nn as nn
from . import callback_register, callback_config_register
from typing import Dict, List
import os
from pytorch_lightning.callbacks import StochasticWeightAveraging


@callback_config_register('weight_average')
class StochasticWeightAveragingCallbackConfig(object):
    """docstring for StochasticWeightAveragingCallbackConfig

        {   //weight_average default
            "_name": "weight_average",
            "config": {
                "swa_epoch_start": 0.8, // swa start epoch
                "swa_lrs": null,
                    //None. Use the current learning rate of the optimizer at the time the SWA procedure starts.
                    //float. Use this value for all parameter groups of the optimizer.
                    //List[float]. A list values for each parameter group of the optimizer.
                "annealing_epochs": 10,
                "annealing_strategy": 'cos',
                "device": null, // save device, null for auto detach, if the gpu is oom, you should change this to 'cpu'
            }
        }
    """
    def __init__(self, config):
        super(StochasticWeightAveragingCallbackConfig, self).__init__()
        config = config['config']
        self.swa_epoch_start = config['swa_epoch_start']
        self.swa_lrs = config["swa_lrs"]
        self.annealing_epochs = config["annealing_epochs"]
        self.annealing_strategy = config["annealing_strategy"]
        self.device = config["device"]

@callback_register('weight_average')
class StochasticWeightAveragingCallback(object):
    """
    """

    def __init__(self, config: StochasticWeightAveragingCallbackConfig):
        super().__init__()
        self.config = config

    def __call__(self, rt_config: Dict):
        """TODO: Docstring for __call__.

        :rt_config: Dict: TODO
        :returns: TODO

        """
        return StochasticWeightAveraging(**self.config.__dict__)
