# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hjson
import pandas as pd
from typing import Union, Dict
from dlk.data.postprocessors import postprocessor_register, postprocessor_config_register, IPostProcessor, IPostProcessorConfig
from dlk.utils.config import ConfigTool
import torch

@postprocessor_config_register('identity')
class IdentityPostProcessorConfig(IPostProcessorConfig):
    """docstring for IdentityPostProcessorConfig
    """

    def __init__(self, config: Dict):
        super(IdentityPostProcessorConfig, self).__init__(config)


@postprocessor_register('identity')
class IdentityPostProcessor(IPostProcessor):
    """docstring for DataSet"""
    def __init__(self, config: IdentityPostProcessorConfig):
        super(IdentityPostProcessor, self).__init__()

    def process(self, stage, outputs, origin_data)->Dict:
        """do nothing except gather the loss
        """
        if 'loss' in outputs:
            return {self.loss_name_map(stage): torch.mean(outputs['loss'])}
        return {}
