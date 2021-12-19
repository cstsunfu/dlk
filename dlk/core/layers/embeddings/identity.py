import torch.nn as nn
from . import embedding_register, embedding_config_register
from typing import Dict, List, Set
from dlk.core.base_module import SimpleModule, BaseModuleConfig
import torch


@embedding_config_register('identity')
class IdentityEmbeddingConfig(BaseModuleConfig):
    """docstring for IdentityEmbeddingConfig
    {
        "config": {
        },
        "_name": "identity",
    }
    """
    def __init__(self, config):
        super(IdentityEmbeddingConfig, self).__init__(config)
        self.post_check(config['config'])


@embedding_register('identity')
class IdentityEmbedding(SimpleModule):
    """
    Sequence labeling model
    """

    def __init__(self, config: IdentityEmbeddingConfig):
        super().__init__(config)
        self.config = config

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """forward
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        return inputs
