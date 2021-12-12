import torch.nn as nn
from . import embedding_register, embedding_config_register
from typing import Dict, List, Set
from dlkit.core.base_module import SimpleModule, BaseModuleConfig
import pickle as pkl
import torch

        
@embedding_config_register('combine_word_char_cnn')
class CombineWordCharCNNEmbeddingConfig(BaseModuleConfig):
    """docstring for BasicModelConfig, this exp. is for static word embedding and cnn char embedding
    {
        "embedding@char": {
            "_base": "combine_word_char_cnn",
        },
        "embedding@word": {
            "_base": "static",
        },
        "config": {
            "word": {
                "embedding_file": "*@*", //the embedding file, must be saved as numpy array by pickle
                "embedding_dim": "*@*",
                "embedding_trace": ".", //default the file itself is the embedding
                "freeze": false, // is freeze
                "padding_idx": 0, //dropout rate
                "output_map": {"embedding": "word_embedding"},
                "input_map": {}, // required_key: provide_key
            },
            "char": {
                "embedding_file": "*@*", //the embedding file, must be saved as numpy array by pickle
                "embedding_dim": 35, //dropout rate
                "embedding_trace": ".", //default the file itself is the embedding
                "freeze": false, // is freeze
                "kernel_sizes": [3], //dropout rate
                "padding_idx": 0,
                "output_map": {"char_embedding": "char_embedding"},
                "input_map": {"char_ids": "char_ids"},
            },
            "dropout": 0, //dropout rate
            "embedding_dim": 135, // this must equal to char.embedding_dim + word.embedding_dim
            "output_map": {"embedding": "embedding"}, // this config do nothing, you can change this
            "input_map": {"char_embedding": "char_embedding", 'word_embedding': "word_embedding"}, // if the output of char and word embedding changed, you also should change this
        },
        "_link":{
            "config.word.embedding_file": ["module@word.config.embedding_file"],
            "config.word.embedding_dim": ["module@word.config.embedding_dim"],
            "config.word.embedding_trace": ["module@word.config.embedding_trace"],
            "config.word.freeze": ["module@word.config.freeze"],
            "config.word.padding_idx": ["module@word.config.padding_idx"],
            "config.word.output_map": ["module@word.config.output_map"],
            "config.word.input_map": ["module@word.config.input_map"],
            "config.char.embedding_file": ["module@char.config.embedding_file"],
            "config.char.embedding_dim": ["module@char.config.embedding_dim"],
            "config.char.embedding_trace": ["module@char.config.embedding_trace"],
            "config.char.freeze": ["module@char.config.freeze"],
            "config.char.kernel_sizes": ["module@char.config.kernel_sizes"],
            "config.char.padding_idx": ["module@char.config.padding_idx"],
            "config.char.output_map": ["module@char.config.output_map"],
            "config.char.input_map": ["module@char.config.input_map"],
        },
        "_name": "combine_word_char_cnn",
    }
    """
    def __init__(self, config: Dict):
        super(CombineWordCharCNNEmbeddingConfig, self).__init__(config)

        self.char_module_name = config['module@char']['_name']
        self.char_config = embedding_config_register(self.char_module_name)(config['module@char'])
        self.word_module_name = config['module@word']['_name']
        self.word_config = embedding_config_register(self.word_module_name)(config['module@word'])
        self.dropout = config['config']['dropout']
        self.embedding_dim = config['config']['embedding_dim']
        assert self.embedding_dim == self.char_config.embedding_dim + self.word_config.embedding_dim, f"The combine embedding_dim must equals to char_embedding+word_embedding, but {self.embedding_dim}!= {self.char_config.embedding_dim+self.word_config.embedding_dim}"

@embedding_register('combine_word_char_cnn')
class CombineWordCharCNNEmbedding(SimpleModule):
    """
    from 'input_ids' generate 'embedding'
    """

    def __init__(self, config: CombineWordCharCNNEmbeddingConfig):
        super().__init__(config)
        self._provided_keys = set() # provided by privous module, will update by the check_keys_are_provided
        self._provide_keys = {'char_embedding'} # provide by this module
        self._required_keys = {'char_ids'} # required by this module
        self.config = config
        self.dropout = nn.Dropout(self.config.dropout)
        self.word_embedding = embedding_register(self.config.word_module_name)(self.config.word_config)
        self.char_embedding = embedding_register(self.config.char_module_name)(self.config.char_config)

        
    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """forward
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        inputs = self.word_embedding(inputs)
        inputs = self.char_embedding(inputs)

        char_embedding = inputs[self.get_input_name("char_embedding")]
        word_embedding = inputs[self.get_input_name("word_embedding")]
        combine_embedding = torch.cat([char_embedding, word_embedding], dim=-1)
        inputs[self.get_output_name('embedding')] = self.dropout(combine_embedding)
        inputs.update(self._logits_gather([inputs[self.get_output_name('embedding')]]))
        return inputs
