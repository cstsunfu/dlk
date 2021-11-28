import torch.nn as nn
from . import encoder_register, encoder_config_register
from typing import Dict, List, Set
from dlkit.core.base_module import SimpleModule, BaseModuleConfig
import torch

        
@encoder_config_register('identity')
class IdentityEncoderConfig(BaseModuleConfig):
    """docstring for IdentityEncoderConfig
    {
        config: {
            output_map: {},
            input_map: {}
        },
        _name: "identity",
    }
    """
    def __init__(self, config):
        super(IdentityEncoderConfig, self).__init__(config)

@encoder_register('identity')
class IdentityEncoder(SimpleModule):
    """
    Sequence labeling model
    """

    def __init__(self, config: IdentityEncoderConfig):
        super().__init__()
        self.config = config

        self._provided_keys = set() # provided by privous module, will update by the check_keys_are_provided
        self._provide_keys = {} # provide by this module
        self._required_keys = {} # required by this module

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """forward
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        return inputs
