# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from functools import partial
from typing import Any, Dict, List

from intc import MISSING, Base, DictField, ListField, NestField, StrField, cregister

from dlk.utils.register import register
from dlk.utils.vocab import Vocabulary

from . import BaseSubProcessor, BaseSubProcessorConfig

logger = logging.getLogger(__name__)


@cregister("subprocessor", "token2id")
class Token2IDConfig(BaseSubProcessorConfig):
    """the token 2 id subprocessor"""

    train_data_set = ListField(
        value=["train", "valid", "test"],
        suggestions=[["train", "valid", "test"]],
        help="the data set should be processed for train stage",
    )
    predict = ListField(
        value=["predict"],
        suggestions=[["predict"]],
        help="the data set should be processed for predict stage",
    )
    online = ListField(
        value=["online"],
        suggestions=[["online"]],
        help="the data set should be processed for online stage",
    )

    class InputMap:
        tokens = StrField(
            value="tokens",
            suggestions=["tokens", "labels"],
            help="the tokens",
        )

    input_map = NestField(
        value=InputMap,
        help="the input map of the processor",
    )

    class OutputMap:
        token_ids = StrField(
            value="token_ids",
            suggestions=["token_ids", "label_ids"],
            help="the token ids",
        )

    output_map = NestField(
        value=OutputMap,
        help="the output map of the processor",
    )
    vocab = StrField(
        value="token_vocab.json",
        suggestions=["token_vocab.json", "label_vocab.json"],
        help="the vocab for the token",
    )


@cregister("subprocessor", "token2id-label")
class Token2IDLabelConfig(Token2IDConfig):
    """the token 2 character id subprocessor, generally used for label to id"""

    predict = ListField(
        value=[],
        suggestions=[["predict"]],
        help="the data set should be processed for predict stage",
    )
    online = ListField(
        value=[],
        suggestions=[["online"]],
        help="the data set should be processed for online stage",
    )

    class InputMap:
        tokens = StrField(
            value="labels",
            suggestions=["tokens", "labels"],
            help="the tokens",
        )

    input_map = NestField(
        value=InputMap,
        help="the input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )

    class OutputMap:
        token_ids = StrField(
            value="label_ids",
            suggestions=["token_ids", "label_ids"],
            help="the token ids",
        )

    output_map = NestField(
        value=OutputMap,
        help="the output map of the processor, the key is the name of the processor provided key, the value is the nexted processor needed key",
    )
    vocab = StrField(
        value="label_vocab.json",
        suggestions=["token_vocab.json", "label_vocab.json"],
        help="the vocab for the token",
    )


@register("subprocessor", "token2id")
class Token2ID(BaseSubProcessor):
    """Use 'Vocabulary' map the tokens to id"""

    def __init__(self, stage: str, config: Token2IDConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.config = config
        self.vocab: Vocabulary = None

    def load_meta(self):
        self.loaded_meta = True
        self.vocab = Vocabulary.load_from_file(
            os.path.join(self.meta_dir, self.config.vocab)
        )

    def process_batch(
        self, batch: Dict[str, List[Any]], deliver_meta: bool = False
    ) -> Dict[str, List[Any]]:
        """Map tokens to IDs using the loaded vocabulary.

        Args:
            batch: Input batch data.
            deliver_meta: Unused.

        Returns:
            Batch with added token_ids column.
        """
        if not self.loaded_meta:
            self.load_meta()

        input_col = self.config.input_map.tokens
        output_col = self.config.output_map.token_ids

        if input_col in batch:
            # Use list comprehension for efficient mapping
            # vocab.auto_get_index handles both single strings and lists of strings
            batch[output_col] = [
                self.vocab.auto_get_index(item) for item in batch[input_col]
            ]

        return batch
