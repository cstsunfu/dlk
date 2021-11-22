import hjson
import pandas as pd
from typing import Union, Dict
from dlkit.utils.parser import BaseConfigParser
from dlkit.data.postprocessors import postprocessor_register, postprocessor_config_register, IPostProcessor
from dlkit.utils.config import ConfigTool

@postprocessor_config_register('identity')
class IdentityPostProcessorConfig(object):
    """docstring for IdentityPostProcessorConfig
    config e.g.
    """

    def __init__(self, config: Dict):
        self.config = config


@postprocessor_register('identity')
class IdentityPostProcessor(IPostProcessor):
    """docstring for DataSet"""
    def __init__(self, config: IdentityPostProcessorConfig):
        super(IdentityPostProcessor, self).__init__()

    def process(self, stage, **inputs)->Dict:
        """TODO: Docstring for process.

        :data: TODO
        :returns: TODO
        """
        return inputs
