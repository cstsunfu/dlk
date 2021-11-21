import torch.nn as nn
from . import embedding_register, embedding_config_register
from typing import Dict, List, Set
from dlkit.core.base_module import SimpleModule
import torch

        
@embedding_config_register('identity')
class IdentityEmbeddingConfig(object):
    """docstring for IdentityEmbeddingConfig
    {
        config: {
            return_logits: "embedding_logits",
            output_map: {},
        },
        _name: "identity",
    }
    """
    def __init__(self, config):
        super(IdentityEmbeddingConfig, self).__init__()
        self.output_map = config['config']['output_map']
        self.return_logits = config['config']['return_logits']
        if self.return_logits:
            raise PermissionError("The identity module not support return logits.")


@embedding_register('identity')
class IdentityEmbedding(SimpleModule):
    """
    Sequence labeling model
    """

    def __init__(self, config: IdentityEmbeddingConfig):
        super().__init__()
        self.config = config

        self._provided_keys = set() # provided by privous module, will update by the check_keys_are_provided
        self._provide_keys = {'embedding'} # provide by this module
        self._required_keys = {'input_ids'} # required by this module

    def provide_keys(self)->Set:
        """TODO: should provide_keys in model?
        """
        return self.set_rename(self._provided_keys.union(self._provide_keys), self.config.output_map)

    def check_keys_are_provided(self, provide: Set[str])->None:
        """
        """
        self._provided_keys = provide
        for required_key in self._required_keys:
            if required_key not in provide:
                raise PermissionError(f"The {self.__class__.__name__} Module required '{required_key}' as input.")

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """forward
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        return self.dict_rename(inputs, self.config.output_map)
