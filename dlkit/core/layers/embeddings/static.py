import torch.nn as nn
from . import embedding_register, embedding_config_register
from typing import Dict, List, Set
from dlkit.core.base_module import SimpleModule
import pickle as pkl
import torch

        
@embedding_config_register('static')
class StaticEmbeddingConfig(object):
    """docstring for BasicModelConfig
    {
        config: {
            embedding_file: "*@*", //the embedding file, must be saved as numpy array by pickle

            //if the embedding_file is a dict, you should provide the dict trace to embedding
            embedding_trace: ".", //default the file itself is the embedding
            /*embedding_trace: "embedding", //this means the <embedding = pickle.load(embedding_file)["embedding"]>*/
            /*embedding_trace: "meta.embedding", //this means the <embedding = pickle.load(embedding_file)['meta']["embedding"]>*/
            freeze: false, // is freeze
            dropout: 0, //dropout rate
            output_map: {},
            return_logits: "embedding_logits",
        },
        _name: "static",
    }
    """
    def __init__(self, config: Dict):
        super(StaticEmbeddingConfig, self).__init__()
        config = config['config']

        embedding_file = config['embedding_file']
        embedding_file = pkl.load(open(embedding_file, 'rb'))
        embedding_trace = config["embedding_trace"]
        if embedding_trace != '.':
            traces = embedding_trace.split('.')
            for trace in traces:
                embedding_file = embedding_file[trace]
        self.embedding = embedding_file
        self.freeze = config['freeze']
        self.dropout = config['dropout']

        self.output_map = config['output_map']
        self.return_logits = config['return_logits']


@embedding_register('static')
class StaticEmbedding(SimpleModule):
    """
    from 'input_ids' generate 'embedding'
    """

    def __init__(self, config: StaticEmbeddingConfig):
        super().__init__()
        self._provided_keys = set() # provided by privous module, will update by the check_keys_are_provided
        self._provide_keys = {'embedding'} # provide by this module
        self._required_keys = {'input_ids'} # required by this module
        self.config = config
        self.dropout = nn.Dropout(self.config.dropout)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(self.config.embedding, dtype=torch.float), freeze=self.config.freeze)
        
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
        inputs['embedding'] = self.dropout(self.embedding(inputs['input_ids']))
        if self.config.return_logits:
            inputs[self.config.return_logits] = inputs['embedding']
        return self.dict_rename(inputs, self.config.output_map)
