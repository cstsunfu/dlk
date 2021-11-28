import torch.nn as nn
from . import embedding_register, embedding_config_register
from typing import Dict, List, Set
from dlkit.core.base_module import SimpleModule, BaseModuleConfig
import torch

        
@embedding_config_register('identity')
class IdentityEmbeddingConfig(BaseModuleConfig):
    """docstring for IdentityEmbeddingConfig
    {
        config: {
            output_map: {},
            input_map: {},
        },
        _name: "identity",
    }
    """
    def __init__(self, config):
        super(IdentityEmbeddingConfig, self).__init__(config)


@embedding_register('identity')
class IdentityEmbedding(SimpleModule):
    """
    Sequence labeling model
    """

    def __init__(self, config: IdentityEmbeddingConfig):
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
