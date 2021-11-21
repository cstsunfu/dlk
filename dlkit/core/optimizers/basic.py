from typing import Dict
import torch.nn as nn
from . import optimizer_register, optimizer_config_register, optimizer_map
import copy


@optimizer_config_register("baisc")
class BasicOptimizerConfig(object):
    """docstring for LinearConfig
    {
        config: {
            optimizer_name: adamw,
            optimizer_config: {
                lr: 1e-3,
                betas: [0.9, 0.999],
                eps: 1e-6,
                weight_decay: 1e-2,
            },
            optimizer_special_groups:[  
            // special paramater groups set to special value, if some config key-value is not set, will use the default config in  optimizer_config. 
            // You should sort the config by priority(
            //     e.g. the first group is ['linear.bias', {weight_decay: 0.1}], the second is [bias, [{weight_decay: 0.2}]], then the weight_decay of "*linea.bias*" will be 0.1, and the weight_decay of others *.bias.* will be 0.2
                ["bias & LayerNorm.bias & LayerNorm.weight", {weight_decay: 0}]
            ]
        },
        _name: "basic",
    }
    """
    def __init__(self, config: Dict):
        super(BasicOptimizerConfig, self).__init__()
        config = config.get('config', {})
        self.optimizer_name = config.get('optimizer_name', 'adamw') # must provide
        self.optimizer_config = config.get('optimizer_config', {})
        self.optimizer_special_groups = config.get('optimizer_special_groups', [])
        

@optimizer_register("basic")
class BasicOptimizer(object):
    def __init__(self, model: nn.Module, config: BasicOptimizerConfig):
        super(BasicOptimizer, self).__init__()
        self.config = config
        self.model = model

    def get_optimizer(self):
        """TODO: Docstring for get_optimizer.
        :returns: TODO
        """
        params = []
        all_named_parameters = list(self.model.named_parameters())
        has_grouped_params = set()
        for special_group in self.config.optimizer_special_groups:
            assert len(special_group) == 2
            key, group_config = copy.deepcopy(special_group)
            keys = [s.strip() for s in key.split('&')]
            group_param = []
            for n, p  in all_named_parameters:
                if n in has_grouped_params:
                    continue
                if any(key in n for key in keys):
                    has_grouped_params.add(n)
                    group_param.append(p)
            group_config['params'] = group_param
            params.append(group_config)

        reserve_params = [p for n, p in all_named_parameters if not n in has_grouped_params]
        params.append({"params": reserve_params})

        return optimizer_map.get(self.config.optimizer_name)(params=params, **self.config.optimizer_config)

    def __call__(self):
        """TODO: Docstring for __call__.
        :returns: TODO

        """
        return self.get_optimizer()
