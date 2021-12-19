import torch.nn as nn
from . import decoder_register, decoder_config_register
from typing import Dict, List, Set
from dlk.core.base_module import SimpleModule, BaseModuleConfig
import torch


@decoder_config_register('identity')
class IdentityDecoderConfig(BaseModuleConfig):
    """docstring for IdentityDecoderConfig
    {
        "config": {
        },
        "_name": "identity",
    }
    """
    def __init__(self, config):
        super(IdentityDecoderConfig, self).__init__(config)
        self.post_check(config['config'])


@decoder_register('identity')
class IdentityDecoder(SimpleModule):
    """
    Sequence labeling model
    """

    def __init__(self, config: IdentityDecoderConfig):
        super().__init__(config)
        self.config = config

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """forward
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        return inputs
