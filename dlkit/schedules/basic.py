from typing import Dict
import torch.nn as nn
from . import schedule_register, schedule_config_register, schedule_map
import copy


@schedule_config_register("baisc")
class BasicScheduleConfig(object):
    """docstring for LinearConfig
    {
        config: {
            schedule_name: linear_warmup,
            schedule_config: {
                num_warmup_steps: 0.1,  # if 0 < num_warmup_steps < 1, it will be set to int(num_training_steps*num_warmup_staps), otherwise set to itself
                # num_training_steps will be set in imodel configure_optimizers
            },
        },
        _name: "basic",
    }
    """
    def __init__(self, config: Dict):
        super(BasicScheduleConfig, self).__init__()
        config = config.get('config', {})
        self.schedule_name = config.get('schedule_name', 'linear_warmup') # must provide
        self.schedule_config = config.get('schedule_config', {})
        

@schedule_register("basic")
class BasicSchedule(object):
    def __init__(self, optimizer: nn.Module, config: BasicScheduleConfig):
        super(BasicSchedule, self).__init__()
        self.config = config
        self.optimizer = optimizer

    def get_schedule(self):
        """TODO: Docstring for get_schedule.
        :returns: TODO
        """
        return schedule_map.get(self.config.schedule_name)(optimizer=self.optimizer, **self.config.schedule_config)

    def __call__(self):
        """TODO: Docstring for __call__.
        :returns: TODO

        """
        return self.get_schedule()
