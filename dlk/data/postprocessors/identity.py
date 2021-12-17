import hjson
import pandas as pd
from typing import Union, Dict
from dlk.utils.parser import BaseConfigParser
from dlk.data.postprocessors import postprocessor_register, postprocessor_config_register, IPostProcessor, IPostProcessorConfig
from dlk.utils.config import ConfigTool
import torch

@postprocessor_config_register('identity')
class IdentityPostProcessorConfig(IPostProcessorConfig):
    """docstring for IdentityPostProcessorConfig
    config e.g.

    """

    def __init__(self, config: Dict):
        super(IdentityPostProcessorConfig, self).__init__(config)


@postprocessor_register('identity')
class IdentityPostProcessor(IPostProcessor):
    """docstring for DataSet"""
    def __init__(self, config: IdentityPostProcessorConfig):
        super(IdentityPostProcessor, self).__init__()

    def process(self, stage, outputs, origin_data)->Dict:
        """TODO: Docstring for process.

        :data: TODO
        :returns: TODO
        """
        if 'loss' in outputs:
            return {self.loss_name_map(stage): torch.sum(outputs['loss'])}
        return {}
