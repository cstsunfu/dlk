# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import pickle as pkl
from typing import Callable, Dict, List, Set

import torch
import torch.nn as nn
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

from dlk.nn.base_module import SimpleModule
from dlk.utils.register import register, register_module_name


@cregister("embedding", "combine_word_char_cnn")
class CombineWordCharCNNEmbeddingConfig(Base):
    """the combine_word_char_cnn embedding module config"""

    char_embedding_dim = IntField(value=MISSING, minimum=0, help="the embedding dim")
    word_embedding_dim = IntField(value=MISSING, minimum=0, help="the embedding dim")
    char_embedding_file = StrField(
        value="char_embedding", help="the char embedding file"
    )
    word_embedding_file = StrField(
        value="word_embedding", help="the word embedding file"
    )
    dropout = FloatField(value=0, minimum=0.0, maximum=1.0, help="dropout rate")

    class InputMap:
        char_ids = StrField(value="char_ids", help="the char ids")
        input_ids = StrField(value="input_ids", help="the input ids")

    input_map = NestField(
        value=InputMap, help="the input map of the combine embedding module"
    )

    class OutputMap:
        embedding = StrField(value="embedding", help="the embedding")

    output_map = NestField(
        value=OutputMap, help="the output map of the combine embedding module"
    )

    submodule = SubModule(
        value={
            "embedding#char": {
                "_base": "static_char_cnn",
                "embedding_file": "@lambda @$$.char_embedding_file",
                "dropout": "@lambda @$$.dropout",
                "embedding_dim": "@lambda @$$.char_embedding_dim",
                "output_map": {"char_embedding": "char_embedding"},
            },
            "embedding#word": {
                "_base": "static",
                "embedding_file": "@lambda @$$.word_embedding_file",
                "dropout": "@lambda @$$.dropout",
                "embedding_dim": "@lambda @$$.word_embedding_dim",
                "output_map": {"embedding": "word_embedding"},
            },
        },
        help="the char and word embedding module",
    )


@register("embedding", "combine_word_char_cnn")
class CombineWordCharCNNEmbedding(SimpleModule):
    """from 'input_ids' and 'char_ids' generate 'embedding'"""

    def __init__(self, config: CombineWordCharCNNEmbeddingConfig):
        super().__init__(config)
        self.dropout = nn.Dropout(float(config.dropout))
        word_embedding_config = config["@embedding#word"]
        self.word_embedding = register.get(
            "embedding", register_module_name(word_embedding_config._module_name)
        )(word_embedding_config)
        char_embedding_config = config["@embedding#char"]
        self.char_embedding = register.get(
            "embedding", register_module_name(char_embedding_config._module_name)
        )(char_embedding_config)
        self.config = config

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        self.word_embedding.init_weight(method)
        self.char_embedding.init_weight(method)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """get the combine char and word embedding

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        inputs = self.word_embedding(inputs)
        inputs = self.char_embedding(inputs)

        combine_embedding = torch.cat(
            [
                inputs[self.char_embedding.config.output_map.char_embedding],
                inputs[self.word_embedding.config.output_map.embedding],
            ],
            dim=-1,
        )
        inputs[self.config.output_map.embedding] = self.dropout(combine_embedding)
        return inputs
