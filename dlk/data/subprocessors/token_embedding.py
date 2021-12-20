"""
Gather tokens embedding from pretrained 'embedding_file' or init embedding(xavier_uniform init, and the range clip in 'bias_clip_range')
The tokens are from 'Tokenizer'(get_vocab) or 'Vocabulary'(word2idx) object(the two must provide only one)
"""
from dlk.utils.vocab import Vocabulary
from dlk.utils.config import BaseConfig, ConfigTool
from typing import Dict, Callable, Set, List
from dlk.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from dlk.utils.logger import Logger
from tokenizers import Tokenizer
import numpy as np

logger = Logger.get_logger()

@subprocessor_config_register('token_embedding')
class TokenEmbeddingConfig(BaseConfig):
    """Config eg.
        {
            "_name": "token_embedding",
            "config": {
                "train": { // only train stage using
                    "embedding_file": "*@*",
                    "tokenizer": null, //List of columns. Every cell must be sigle token or list of tokens or set of tokens
                    "vocab": null,
                    "deliver": "token_embedding", // output Vocabulary object (the Vocabulary of labels) name.
                    "embedding_size": 200,
                    "bias_clip_range": [0.5, 0.1], // the init embedding bias weight range, if you provide two, the larger is the up bound the lower is low bound; if you provide one value, we will use it as the bias
                }
            }
        }
    """

    def __init__(self, stage, config):
        super(TokenEmbeddingConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        if not self.config:
            return
        self.embedding_file = self.config.get("embedding_file")
        self.tokenizer = self.config.get("tokenizer")
        self.bias_clip_range = self.config['bias_clip_range']
        self.vocab = self.config['vocab']
        self.deliver = self.config.get("deliver")
        self.embedding_size = self.config["embedding_size"]
        self.post_check(self.config, used=[
            "embedding_file",
            "tokenizer",
            "vocab",
            "deliver",
            "embedding_size",
            "bias_clip_range",
        ])

@subprocessor_register('token_embedding')
class TokenEmbedding(ISubProcessor):
    """
    """
    def __init__(self, stage: str, config: TokenEmbeddingConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        if not self.config.config:
            logger.info(f"Skip 'token_embedding' at stage {self.stage}")
            return
        if config.tokenizer:
            self.tokenizer = Tokenizer.from_file(config.tokenizer)
        else:
            self.tokenizer = None
        if config.embedding_file:
            self.origin_embedding = self.get_embedding(config.embedding_file, config.embedding_size)
        else:
            self.origin_embedding = {}

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
                if len(sp_line)<=embedding_size:
                    logger.warning(f"The {i}th line len： {len(sp_line)}, token is {sp_line[0]}")
                    continue
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
        without_embedding_tokens = 0
        fuzzy_match_tokens = 0
        bias = 0.1
        if len(self.config.bias_clip_range) == 1:
            bias = self.config.bias_clip_range[0]
        else:
            assert len(self.config.bias_clip_range) == 2, "You must provide the clip range, one or two value"
            low:float = min(self.config.bias_clip_range)
            up:float = max(self.config.bias_clip_range)
            # xavier_uniform_ init method
            bias:float = np.sqrt(6.0 / (len(vocab) + self.config.embedding_size))
            if bias>up:
                bias = up
            elif bias<low:
                bias = low
            # bias = np.sqrt(3/self.config.embedding_size)
        for token in vocab:
            if token not in embedding_dict:
                if (token.lower() not in embedding_dict) and (token.upper() not in embedding_dict):
                    embedding_dict[token] = list(np.random.uniform(-bias, bias, self.config.embedding_size))
                    without_embedding_tokens += 1
                else:
                    fuzzy_match_tokens += 1
                    if token.lower() in embedding_dict:
                        embedding_dict[token] = embedding_dict[token.lower()]
                    else:
                        embedding_dict[token] = embedding_dict[token.upper()]
        logger.info(f"All tokens num is {len(vocab)}, fuzzy mathing(lower or upper match) num is {fuzzy_match_tokens}, OOV token num is {without_embedding_tokens}")
        return embedding_dict

    def process(self, data: Dict)->Dict:
        if not self.config.config:
            return data
        if self.tokenizer is not None and self.config.vocab:
            raise PermissionError(f"The tokenizer and vocab must provide one.")
        if self.tokenizer:
            token2id = self.tokenizer.get_vocab()
            id2token = {value:key for key,value in token2id.items()}
        else:
            assert self.config.vocab, f"The tokenizer and vocab must provide one."
            vocab = data[self.config.vocab]
            token2id = vocab['word2idx']
            id2token = vocab['idx2word']
        embedding_dict = self.update_embedding(self.origin_embedding, token2id)
        embedding_mat = [embedding_dict[id2token[id]] for id in range(len(id2token))]
        data[self.config.deliver] = np.array(embedding_mat)
        return data
