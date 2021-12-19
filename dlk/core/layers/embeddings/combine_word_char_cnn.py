import torch.nn as nn
from . import embedding_register, embedding_config_register
from typing import Callable, Dict, List, Set
from dlk.core.base_module import SimpleModule, BaseModuleConfig
import pickle as pkl
import torch


@embedding_config_register('combine_word_char_cnn')
class CombineWordCharCNNEmbeddingConfig(BaseModuleConfig):
    """docstring for BasicModelConfig, this exp. is for static word embedding and cnn char embedding
    {
        "_name": "combine_word_char_cnn",
        "embedding@char": {
            "_base": "static_char_cnn",
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
            "embedding_dim": "*@*", // this must equal to char.embedding_dim + word.embedding_dim
            "output_map": {"embedding": "embedding"}, // this config do nothing, you can change this
            "input_map": {"char_embedding": "char_embedding", 'word_embedding': "word_embedding"}, // if the output of char and word embedding changed, you also should change this
        },
        "_link":{
            "config.word.embedding_file": ["embedding@word.config.embedding_file"],
            "config.word.embedding_dim": ["embedding@word.config.embedding_dim"],
            "config.word.embedding_trace": ["embedding@word.config.embedding_trace"],
            "config.word.freeze": ["embedding@word.config.freeze"],
            "config.word.padding_idx": ["embedding@word.config.padding_idx"],
            "config.word.output_map": ["embedding@word.config.output_map"],
            "config.word.input_map": ["embedding@word.config.input_map"],
            "config.char.embedding_file": ["embedding@char.config.embedding_file"],
            "config.char.embedding_dim": ["embedding@char.config.embedding_dim"],
            "config.char.embedding_trace": ["embedding@char.config.embedding_trace"],
            "config.char.freeze": ["embedding@char.config.freeze"],
            "config.char.kernel_sizes": ["embedding@char.config.kernel_sizes"],
            "config.char.padding_idx": ["embedding@char.config.padding_idx"],
            "config.char.output_map": ["embedding@char.config.output_map"],
            "config.char.input_map": ["embedding@char.config.input_map"],
        },
    }
    """
    def __init__(self, config: Dict):
        super(CombineWordCharCNNEmbeddingConfig, self).__init__(config)

        self.char_module_name = config['embedding@char']['_name']
        self.char_config = embedding_config_register[self.char_module_name](config['embedding@char'])
        self.word_module_name = config['embedding@word']['_name']
        self.word_config = embedding_config_register[self.word_module_name](config['embedding@word'])
        self.dropout = config['config']['dropout']
        self.embedding_dim = config['config']['embedding_dim']
        assert self.embedding_dim == self.char_config.embedding_dim + self.word_config.embedding_dim, f"The combine embedding_dim must equals to char_embedding+word_embedding, but {self.embedding_dim}!= {self.char_config.embedding_dim+self.word_config.embedding_dim}"

        self.post_check(config['config'], used=[
            "word.embedding_file",
            "word.embedding_dim",
            "word.embedding_trace",
            "word.freeze",
            "word.padding_idx",
            "word.output_map",
            "word.input_map",
            "char.embedding_file",
            "char.embedding_dim",
            "char.embedding_trace",
            "char.freeze",
            "char.padding_idx",
            "char.output_map",
            "char.input_map",
            "char.kernel_sizes",
            "dropout",
            "embedding_dim",
        ])


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
        self.dropout = nn.Dropout(float(self.config.dropout))
        self.word_embedding = embedding_register[self.config.word_module_name](self.config.word_config)
        self.char_embedding = embedding_register[self.config.char_module_name](self.config.char_config)

    def init_weight(self, method: Callable):
        """TODO: Docstring for init_weight.
        :arg1: TODO
        :returns: TODO
        """
        self.word_embedding.init_weight(method)
        self.char_embedding.init_weight(method)

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
        if self._logits_gather.layer_map:
            inputs.update(self._logits_gather([inputs[self.get_output_name('embedding')]]))
        return inputs
