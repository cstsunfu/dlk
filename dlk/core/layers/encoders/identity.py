import torch.nn as nn
from . import encoder_register, encoder_config_register
from typing import Dict, List, Set
from dlk.core.base_module import SimpleModule, BaseModuleConfig
import torch


@encoder_config_register('identity')
class IdentityEncoderConfig(BaseModuleConfig):
    """docstring for IdentityEncoderConfig
    {
        "config": {
        },
        "_name": "identity",
    }
    """
    def __init__(self, config):
        super(IdentityEncoderConfig, self).__init__(config)
        self.post_check(config['config'])

@encoder_register('identity')
class IdentityEncoder(SimpleModule):
    """
    Sequence labeling model
    """

    def __init__(self, config: IdentityEncoderConfig):
        super().__init__(config)
        self.config = config

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """forward
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        return inputs
