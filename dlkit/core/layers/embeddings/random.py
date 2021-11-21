from . import embedding_register, embedding_config_register
from typing import Dict, List
from dlkit.core.base_module import SimpleModule
import pickle as pkl
import torch.nn as nn
import torch
import numpy as np

        
@embedding_config_register('random')
class RandomEmbeddingConfig(object):
    """docstring for BasicModelConfig
    {
        config: {
            vocab_size: "*@*",
            embedding_dim: "*@*",
            dropout: 0, //dropout rate
        },
        _name: "random",
    }
    """
    def __init__(self, config: Dict):
        super(RandomEmbeddingConfig, self).__init__()
        config = config.get('config', {})
        self.vocab_size = config.get('vocab_size', 1)
        self.embedding_dim = config.get('embedding_dim', 1)
        self.dropout = config.get('dropout', 0.0)


@embedding_register('random')
class RandomEmbedding(SimpleModule):
    """
    from 'input_ids' generate 'embedding'
    """

    def __init__(self, config: RandomEmbeddingConfig):
        super().__init__()
        self._provided_keys = [] # provided by privous module, will update by the check_keys_are_provided
        self._provide_keys = ['embedding'] # provide by this module
        self._required_keys = ['input_ids'] # required by this module
        self.config = config
        self.dropout = nn.Dropout(self.config.dropout)
        normal =  torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([2.0/self.config.embedding_dim]))
        self.embedding = nn.Embedding.from_pretrained(normal.sample((self.config.vocab_size, self.config.embedding_dim)))
        
    def provide_keys(self):
        """TODO: should provide_keys in model?
        """
        if self.provide_keys:
            return self._provided_keys + self._provide_keys
        return self._provide_keys

    def check_keys_are_provided(self, provide: List[str])->None:
        """TODO: should check keys in model?
        """
        self._provided_keys = provide
        for required_key in self._required_keys:
            if required_key not in provide:
                raise PermissionError(f"The RandomEmbedding Module required 'input_ids' as input. You should explicit provide the provided keys (list[str]) for check.")

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """forward
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        inputs['embedding'] = self.dropout(self.embedding(inputs['input_ids']))

        return inputs
