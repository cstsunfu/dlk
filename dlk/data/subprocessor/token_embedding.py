# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from functools import partial
from typing import Callable, Dict, List, Set

import numpy as np
import pandas as pd
from intc import (
    MISSING,
    AnyField,
    Base,
    BoolField,
    DictField,
    FloatField,
    IntField,
    ListField,
    NestField,
    StrField,
    SubModule,
    cregister,
)
from tokenizers import Tokenizer

from dlk.utils.io import open
from dlk.utils.register import register
from dlk.utils.vocab import Vocabulary

from . import BaseSubProcessor, BaseSubProcessorConfig

logger = logging.getLogger(__name__)


@cregister("subprocessor", "token_embedding")
class TokenEmbeddingConfig(BaseSubProcessorConfig):
    """the token embedding subprocessor"""

    train_data_set = ListField(
        value=["train"],
        suggestions=[["train", "valid", "test"]],
        help="the data set should be processed for train stage",
    )
    embedding_file = StrField(
        value=MISSING,
        suggestions=["embedding_file"],
        help="the embedding file path",
    )
    tokenizer_path = StrField(
        value=None,
        suggestions=["tokenizer"],
        additions=[None],
        help="the tokenizer path, when this is provided, we will use the tokenizer to get the vocab, not effective by `meta_dir`",
    )
    vocab = StrField(
        value=None,
        suggestions=["vocab"],
        additions=[None],
        help="the vocab path(dump by `Vocabulary`), when this is provided",
    )
    token_embedding = StrField(
        value="token_embedding",
        suggestions=["token_embedding"],
        help="the embedding saved name",
    )
    embedding_size = IntField(value=MISSING, minimum=1, help="the embedding size")

    class BiasClipRange:
        lower = FloatField(
            value=-10e9,
            help="the init embedding bias weight range lower bound",
        )
        upper = FloatField(
            value=10e9,
            help="the init embedding bias weight range upper bound",
        )

    bias_clip_range = NestField(
        value=BiasClipRange, help="the init embedding bias weight range"
    )


@register("subprocessor", "token_embedding")
class TokenEmbedding(BaseSubProcessor):
    """
    Gather tokens embedding from pretrained 'embedding_file' or init embedding(xavier_uniform init, and the range clip in 'bias_clip_range')

    The tokens are from 'Tokenizer'(get_vocab) or 'Vocabulary'(word2idx) object(the two must provide only one)
    """

    def __init__(self, stage: str, config: TokenEmbeddingConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.stage = stage
        self.config = config
        self.tokenizer: Tokenizer = None
        self.vocab: Vocabulary = None
        if self.config.embedding_file:
            self.origin_embedding = self.get_embedding(
                self.config.embedding_file, self.config.embedding_size
            )
        else:
            self.origin_embedding = {}

    def load_meta(self):
        self.loaded_meta = True
        if self.config.vocab:
            self.vocab = Vocabulary.load_from_file(
                os.path.join(self.meta_dir, self.config.vocab)
            )
        if self.config.tokenizer_path:
            with open(self.config.tokenizer_path, "r", encoding="utf-8") as f:
                tokenizer_str = json.dumps(json.load(f))
            self.tokenizer = Tokenizer.from_str(tokenizer_str)

    def get_embedding(self, file_path, embedding_size) -> Dict[str, List[float]]:
        """load the embeddings from file_path, and only get the last embedding_size dimensions embedding

        Args:
            file_path: embedding file path
            embedding_size: the embedding dim

        Returns:
            >>> embedding_dict
            >>> {
            >>>     "word": [embedding, ...]
            >>> }
        """
        embedding_dict = {}
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # if the first line is statistic info, continue
                if i == 0 and len(line.split()) <= embedding_size:
                    continue
                sp_line = line.split()
                if len(sp_line) <= embedding_size:
                    logger.warning(
                        f"The {i}th line len： {len(sp_line)}, token is {sp_line[0]}"
                    )
                    continue
                word = sp_line[0]
                vector = list(map(float, sp_line[-embedding_size:]))
                embedding_dict[word] = vector
        return embedding_dict

    def update_embedding(self, embedding_dict: Dict[str, List[float]], vocab: Dict):
        """update the embedding_dict which token in vocab but not in embedding_dict

        Args:
            embedding_dict: word->embedding dict
            vocab: token vocab

        Returns:
            updated embedding_dict

        """
        without_embedding_tokens = 0
        fuzzy_match_tokens = 0
        low: float = self.config.bias_clip_range.lower
        up: float = self.config.bias_clip_range.upper
        # xavier_uniform_ init method
        bias: float = np.sqrt(6.0 / (len(vocab) + self.config.embedding_size))
        if bias > up:
            bias = up
        elif bias < low:
            bias = low
            # bias = np.sqrt(3/self.config.embedding_size)
        for token in vocab:
            if token not in embedding_dict:
                if (token.lower() not in embedding_dict) and (
                    token.upper() not in embedding_dict
                ):
                    embedding_dict[token] = list(
                        np.random.uniform(-bias, bias, self.config.embedding_size)
                    )
                    without_embedding_tokens += 1
                else:
                    fuzzy_match_tokens += 1
                    if token.lower() in embedding_dict:
                        embedding_dict[token] = embedding_dict[token.lower()]
                    else:
                        embedding_dict[token] = embedding_dict[token.upper()]
        logger.info(
            f"All tokens num is {len(vocab)}, fuzzy matching(lower or upper match) num is {fuzzy_match_tokens}, OOV token num is {without_embedding_tokens}"
        )
        return embedding_dict

    def process(self, data: pd.DataFrame, deliver_meta: bool) -> pd.DataFrame:
        """Character gather entry

        Args:
            data:
            >>> |sentence |label|
            >>> |---------|-----|
            >>> |sent_a...|la   |
            >>> |sent_b...|lb   |

            deliver_meta:
                if there are some meta info need to deliver to next processor, and deliver_meta is True, save the meta info to datadir
        Returns:
            processed data

        """
        if not deliver_meta:
            return data

        if not self.loaded_meta:
            self.load_meta()

        if self.tokenizer is not None and self.vocab:
            raise PermissionError(f"The tokenizer and vocab must provide one.")
        if self.tokenizer:
            token2id: Dict = self.tokenizer.get_vocab()
            id2token: Dict = {value: key for key, value in token2id.items()}
        else:
            token2id: Dict = self.vocab.word2idx
            id2token: Dict = self.vocab.idx2word
        embedding_dict = self.update_embedding(self.origin_embedding, token2id)
        embedding_mat = np.array(
            [embedding_dict[id2token[id]] for id in range(len(id2token))]
        )

        with open(os.path.join(self.meta_dir, self.config.token_embedding), "wb") as f:
            embedding_mat.dump(f)

        return data
