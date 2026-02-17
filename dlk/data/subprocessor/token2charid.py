# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from functools import partial
from typing import Any, Dict, List

from intc import Base, IntField, ListField, NestField, StrField, cregister

from dlk.utils.register import register
from dlk.utils.vocab import Vocabulary

from . import BaseSubProcessor, BaseSubProcessorConfig

logger = logging.getLogger(__name__)


@cregister("subprocessor", "token2charid")
class Token2CharIDConfig(BaseSubProcessorConfig):
    """the token 2 character id subprocessor"""

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
        sentence = StrField(value="sentence", help="the sentence")
        offsets = StrField(value="offsets", help="the offsets")

    input_map = NestField(value=InputMap, help="input map")

    class OutputMap:
        char_ids = StrField(value="char_ids", help="the char ids")

    output_map = NestField(value=OutputMap, help="output map")

    vocab = StrField(value="char_vocab.json", help="the vocab file for the character")
    max_token_len = IntField(
        value=20,
        minimum=1,
        help="the max length of token characters",
    )


@register("subprocessor", "token2charid")
class Token2CharID(BaseSubProcessor):
    """Use 'Vocabulary' map the character from tokens to id."""

    def __init__(self, stage: str, config: Token2CharIDConfig, meta_dir: str):
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
        """Process batch to generate character IDs for tokens.

        Args:
            batch: Input batch.
            deliver_meta: Unused.

        Returns:
            Batch with char_ids.
        """
        if not self.loaded_meta:
            self.load_meta()

        sent_col = self.config.input_map.sentence
        offset_col = self.config.input_map.offsets
        output_col = self.config.output_map.char_ids
        max_len = self.config.max_token_len
        pad_id = self.vocab.get_index(self.vocab.pad)

        if sent_col in batch and offset_col in batch:
            batch_char_ids = []

            # Iterate over batch samples
            for sentence, offsets in zip(batch[sent_col], batch[offset_col]):
                sample_char_ids = []
                for start, end in offsets:
                    # Extract token text
                    if start == end:
                        token_text = ""
                    else:
                        token_text = sentence[start:end]

                    # Truncate
                    token_chars = list(token_text[:max_len])

                    # Convert to IDs
                    ids = [self.vocab.get_index(c) for c in token_chars]

                    # Pad
                    if len(ids) < max_len:
                        ids.extend([pad_id] * (max_len - len(ids)))

                    sample_char_ids.append(ids)
                batch_char_ids.append(sample_char_ids)

            batch[output_col] = batch_char_ids

        return batch
