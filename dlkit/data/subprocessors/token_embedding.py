from dlkit.utils.vocab import Vocabulary
from dlkit.utils.config import ConfigTool
from typing import Dict, Callable, Set, List
from dlkit.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from tokenizers import Tokenizer
import numpy as np


@subprocessor_config_register('token_embedding')
class TokenEmbeddingConfig(object):
    """Config eg.
        {
            "_name": "token_embedding",
            "config": {
                "train": { // only train stage using
                    "embedding_file": "*@*",
                    "tokenizer": "*@*", //List of columns. Every cell must be sigle token or list of tokens or set of tokens
                    "deliver": "token_embedding", // output Vocabulary object (the Vocabulary of labels) name. 
                    "embedding_size": 200,
                }
            }
        }
    """

    def __init__(self, stage, config):
        self.config = ConfigTool.get_config_by_stage(stage, config)

        self.embedding_file = self.config.get("embedding_file")
        self.tokenizer = self.config.get("tokenizer")
        self.deliver = self.config.get("deliver")
        self.embedding_size = self.config.get("embedding_size")

@subprocessor_register('token_embedding')
class TokenEmbedding(ISubProcessor):
    """
    """
    def __init__(self, stage: str, config: TokenEmbeddingConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.tokenizer = Tokenizer.from_file(config.tokenizer)
        self.origin_embedding = self.get_embedding(config.embedding_file, config.embedding_size)

    def get_embedding(self, file_path, embedding_size)->Dict[str, List[float]]:
        """TODO: Docstring for get_embedding.
        """
        embedding_dict = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # if the first line is statistic info, continue
                if i==0 and len(line.split())<=embedding_size:
                    continue
                sp_line = line.split()
                assert len(sp_line)>=embedding_size+1
                word = sp_line[0]
                vector = list(map(float, sp_line[-embedding_size:]))
                embedding_dict[word] = vector
        return embedding_dict

    def update_embedding(self, embedding_dict, vocab):
        """init a embedding to the embedding_dict when some token in vocab but not in embedding_dict

        :embedding_dict: Dict
        :vocab: List[str]
        :returns: updated embedding_dict
        """
        for token in vocab:
            if token not in embedding_dict:
                embedding_dict[token] = [np.random.normal(scale=2.0/self.config.embedding_size) for _ in range(self.config.embedding_size)]
        return embedding_dict

    def process(self, data: Dict)->Dict:
        token2id = self.tokenizer.get_vocab()
        id2token = {value:key for key,value in token2id.items()}
        embedding_dict = self.update_embedding(self.origin_embedding, token2id)
        embedding_mat = [embedding_dict[id2token[id]] for id in range(len(id2token))]
        data[self.config.deliver] = np.array(embedding_mat)
        return data
