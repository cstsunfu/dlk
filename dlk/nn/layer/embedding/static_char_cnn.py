# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pickle as pkl
from typing import Dict, List, Set

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
from dlk.utils.io import open
from dlk.utils.register import register, register_module_name

logger = logging.getLogger(__name__)


@cregister("embedding", "static_char_cnn")
class StaticCharCNNEmbeddingConfig(Base):
    """
    the static_char_cnn embedding module config
    """

    embedding_file = StrField(
        value=MISSING,
        help="the embedding file, must be saved as numpy array by pickle",
    )
    embedding_dim = IntField(value=MISSING, minimum=0, help="the embedding dim")
    freeze = BoolField(value=False, help="whether to freeze the embedding")
    dropout = FloatField(value=0, minimum=0.0, maximum=1.0, help="dropout rate")
    padding_idx = IntField(value=0, minimum=0, help="padding index")
    kernel_sizes = ListField(
        value=[3],
        help="the kernel sizes of the cnn",
        validator=lambda x: all([i > 0 for i in x]),
    )

    class InputMap:
        char_ids = StrField(value="char_ids", help="the input name of char ids")

    input_map = NestField(
        value=InputMap,
        help="the input map of the char embedding module",
    )

    class OutputMap:
        char_embedding = StrField(
            value="char_embedding", help="the output name of char embedding"
        )

    output_map = NestField(
        value=OutputMap,
        help="the output map of the char embedding module",
    )

    submodule = SubModule(
        {
            "module": {
                "_base": "conv1d",
                "in_channels": "@$$.embedding_dim @lambda x: x",
                "out_channels": "@$$.embedding_dim @lambda x: x",
                "kernel_sizes": "@$$.kernel_sizes @lambda x: x",
            }
        },
        help="the cnn module of the char embedding",
    )


@register("embedding", "static_char_cnn")
class StaticCharCNNEmbedding(SimpleModule):
    """from 'char_ids' generate 'embedding'"""

    def __init__(self, config: StaticCharCNNEmbeddingConfig):
        super().__init__(config)
        self.config = config
        embedding_file = self.config.embedding_file
        with open(embedding_file, "rb") as f:
            embedding_file = pkl.load(f)
        cnn_configs = config._get_modules("module")
        assert len(cnn_configs) == 1
        self.cnn = register.get(
            "module", register_module_name(cnn_configs[0]._module_name)
        )(cnn_configs[0])

        self.dropout = nn.Dropout(float(self.config.dropout))
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_file, dtype=torch.float),
            freeze=self.config.freeze,
            padding_idx=0,
        )
        assert self.embedding.weight.shape[-1] == self.config.embedding_dim

    def init_weight(self, method):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        self.cnn.init_weight(method)
        logger.info(f"The static embedding is loaded the pretrained.")

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """fit the char embedding to cnn and pool to word_embedding

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        char_ids = inputs[self.config.input_map.char_ids]
        char_mask = (char_ids == 0).bool()
        char_embedding = self.embedding(char_ids)
        bs, seq_len, token_len, emb_dim = char_embedding.shape
        char_embedding = char_embedding.view(bs * seq_len, token_len, emb_dim)
        char_embedding = char_embedding.transpose(1, 2)
        char_embedding = self.cnn(char_embedding)  # bs*seq_len, emb_dim, token_len
        word_embedding = (
            char_embedding.masked_fill_(
                char_mask.view(bs * seq_len, 1, token_len), -1000
            )
            .max(-1)[0]
            .view(bs, seq_len, emb_dim)
            .contiguous()
        )

        inputs[self.config.output_map.char_embedding] = self.dropout(word_embedding)
        return inputs
