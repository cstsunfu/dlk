# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional

from intc import (
    MISSING,
    Base,
    BoolField,
    IntField,
    ListField,
    NestField,
    StrField,
    cregister,
)

from dlk.utils.register import register
from dlk.utils.tokenizer_util import load_fast_tokenizer

from . import BaseSubProcessor, BaseSubProcessorConfig

logger = logging.getLogger(__name__)


@cregister("subprocessor", "fast_tokenizer")
class FastTokenizerConfig(BaseSubProcessorConfig):
    """
    FastTokenizer using HuggingFace transformers AutoTokenizer.
    Wraps the underlying rust-based tokenizers for speed and compatibility.
    """

    train_data_set = ListField(
        value=["train", "valid", "test"],
        suggestions=[["train", "valid", "test"]],
        help="the data set should be processed for train stage",
    )
    predict_data_set = ListField(
        value=["predict"],
        suggestions=[["predict"], []],
        help="the data set should be processed for predict stage",
    )
    online_data_set = ListField(
        value=["online"],
        suggestions=[["online"], []],
        help="the data set should be processed for online stage",
    )

    class InputMap:
        sentence = StrField(
            value="sentence",
            help="for single input, tokenize the 'sentence' column",
        )
        sentence_a = StrField(
            value="sentence_a",
            help="for pair inputs, tokenize the 'sentence_a' column",
        )
        sentence_b = StrField(
            value="sentence_b",
            help="for pair inputs, tokenize the 'sentence_b' column",
        )
        pretokenized_words = StrField(
            value="pretokenized_words",
            help="pretokenized word list related to sentence",
        )
        pretokenized_words_a = StrField(
            value="pretokenized_words_a",
            help="pretokenized word list a related to sentence_a",
        )
        pretokenized_words_b = StrField(
            value="pretokenized_words_b",
            help="pretokenized word list b related to sentence_b",
        )
        pretokenized_word_offsets = StrField(
            value="pretokenized_word_offsets",
            help="pretokenized word offsets for fix offset",
        )
        pretokenized_word_offsets_a = StrField(
            value="pretokenized_word_offsets_a",
            help="pretokenized word offsets for fix offset",
        )
        pretokenized_word_offsets_b = StrField(
            value="pretokenized_word_offsets_b",
            help="pretokenized word offsets for fix offset",
        )

    input_map = NestField(value=InputMap, help="the input map of the processor")

    class OutputMap:
        tokens = StrField(value="tokens", help="the output tokens list")
        ids = StrField(value="input_ids", help="the output input_ids")
        attention_mask = StrField(
            value="attention_mask", help="the output attention_mask"
        )
        type_ids = StrField(value="type_ids", help="the output token_type_ids")
        special_tokens_mask = StrField(
            value="special_tokens_mask", help="the output special_tokens_mask"
        )
        overflowing = StrField(
            value="overflowing", help="the output overflowing tokens"
        )
        offsets = StrField(value="offsets", help="the output offsets mapping")
        word_ids = StrField(value="word_ids", help="the output word_ids mapping")
        sequence_ids = StrField(value="sequence_ids", help="the output sequence_ids")

    output_map = NestField(value=OutputMap, help="the output map of the processor")

    tokenizer_path = StrField(
        value=MISSING,
        help="The path to the pretrained model directory (containing tokenizer.json, config.json, etc.) or a specific tokenizer file.",
    )

    class Truncation:
        stride = IntField(value=0, help="the stride for truncation/sliding window")
        max_length = IntField(
            value=512, minimum=1, help="the max length for truncation"
        )
        strategy = StrField(
            value="longest_first",
            options=["longest_first", "only_first", "only_second", "do_not_truncate"],
            help="the truncation strategy. If 'do_not_truncate', truncation is disabled.",
        )

    truncation = NestField(value=Truncation)

    class ProcessData:
        is_pretokenized = BoolField(
            value=False,
            help="whether the input is already split into words (list of strings)",
        )
        add_special_tokens = BoolField(
            value=True, help="whether to add special tokens (CLS, SEP, etc.)"
        )

    process_data = NestField(value=ProcessData)

    expand_examples = BoolField(
        value=False,
        help="If True, return overflowing tokens as new examples (e.g. for sliding window). If False, overflowing tokens are discarded or returned in 'overflowing' column.",
    )
    input_type = StrField(
        value="single",
        options=["single", "pair"],
        help="the input type of the tokenizer, single or pair",
    )
    fix_offset = BoolField(
        value=False,
        help="Whether fix the offset for the pretokenized word. If True, requires `pretokenized_word_offsets` input.",
    )


@register("subprocessor", "fast_tokenizer")
class FastTokenizer(BaseSubProcessor):
    """
    FastTokenizer wrapper using `transformers.AutoTokenizer`.

    Features:
    - Supports loading from HuggingFace Hub or local directory.
    - Supports single sentence and sentence pair tokenization.
    - Supports pre-tokenized inputs (is_split_into_words=True).
    - Supports sliding window / example expansion (return_overflowing_tokens).
    - Automatically handles offsets mapping and word_ids.
    """

    def __init__(self, stage: str, config: FastTokenizerConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.config = config
        self.tokenizer = load_fast_tokenizer(self.config.tokenizer_path)

    def _fix_offset_one(self, offset, word_id, type_id, word_offsets_a, word_offsets_b):
        """
        Fix offsets when input is pre-tokenized.
        Transformers tokenizer returns offsets relative to the word start when is_split_into_words=True.
        We need to shift them by the word's start position in the original sentence.
        """
        if offset == (0, 0):
            return offset

        current_word_offsets = word_offsets_a if type_id == 0 else word_offsets_b

        if (
            word_id is None
            or current_word_offsets is None
            or word_id >= len(current_word_offsets)
        ):
            return offset

        word_start = current_word_offsets[word_id][0]
        return (offset[0] + word_start, offset[1] + word_start)

    def process_batch(
        self, batch: Dict[str, List[Any]], deliver_meta: bool = False
    ) -> Dict[str, List[Any]]:
        """
        Process a batch of text.

        Args:
            batch: Dict with keys like 'sentence', 'sentence_a', etc.

        Returns:
            Dict with 'input_ids', 'attention_mask', etc.
            If expand_examples is True, the number of rows may increase.
        """

        # 1. Gather Inputs
        if self.config.input_type == "single":
            if self.config.process_data.is_pretokenized:
                inputs = batch[self.config.input_map.pretokenized_words]
            else:
                inputs = batch[self.config.input_map.sentence]
            inputs_pair = None
        else:  # pair
            if self.config.process_data.is_pretokenized:
                inputs = batch[self.config.input_map.pretokenized_words_a]
                inputs_pair = batch[self.config.input_map.pretokenized_words_b]
            else:
                inputs = batch[self.config.input_map.sentence_a]
                inputs_pair = batch[self.config.input_map.sentence_b]

        # 2. Configure Truncation
        truncation = True
        if self.config.truncation.strategy == "do_not_truncate":
            truncation = False

        # 3. Call Transformers Tokenizer
        encodings = self.tokenizer(
            text=inputs,
            text_pair=inputs_pair,
            is_split_into_words=self.config.process_data.is_pretokenized,
            add_special_tokens=self.config.process_data.add_special_tokens,
            padding=False,
            truncation=truncation,
            max_length=self.config.truncation.max_length,
            stride=self.config.truncation.stride,
            return_overflowing_tokens=self.config.expand_examples,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            verbose=False,
        )

        # 4. Prepare Output
        output_map = self.config.output_map
        result = {
            output_map.ids: encodings["input_ids"],
            output_map.attention_mask: encodings["attention_mask"],
            output_map.special_tokens_mask: encodings["special_tokens_mask"],
            output_map.offsets: encodings["offset_mapping"],
        }

        # Handle optional keys
        if "token_type_ids" in encodings:
            result[output_map.type_ids] = encodings["token_type_ids"]
        else:
            result[output_map.type_ids] = [
                [0] * len(ids) for ids in encodings["input_ids"]
            ]

        if output_map.tokens and output_map.tokens != "tokens":
            result[output_map.tokens] = [
                encodings.tokens(i) for i in range(len(encodings["input_ids"]))
            ]
        else:
            result[output_map.tokens] = [
                self.tokenizer.convert_ids_to_tokens(ids)
                for ids in encodings["input_ids"]
            ]

        result[output_map.word_ids] = [
            encodings.word_ids(i) for i in range(len(encodings["input_ids"]))
        ]
        result[output_map.sequence_ids] = [
            encodings.sequence_ids(i) for i in range(len(encodings["input_ids"]))
        ]

        # 5. Handle Expansion
        if self.config.expand_examples:
            sample_map = encodings.get("overflow_to_sample_mapping")
            if sample_map is None:
                sample_map = list(range(len(encodings["input_ids"])))

            expanded_batch = {}
            for key, values in batch.items():
                expanded_batch[key] = [values[i] for i in sample_map]

            expanded_batch.update(result)
            batch = expanded_batch
        else:
            batch.update(result)

        # 6. Fix Offsets
        if self.config.process_data.is_pretokenized and self.config.fix_offset:
            pre_offsets_a = batch.get(self.config.input_map.pretokenized_word_offsets_a)
            pre_offsets_b = batch.get(self.config.input_map.pretokenized_word_offsets_b)
            pre_offsets_single = batch.get(
                self.config.input_map.pretokenized_word_offsets
            )

            fixed_offsets_list = []
            for i, (offsets, word_ids, type_ids) in enumerate(
                zip(
                    batch[output_map.offsets],
                    batch[output_map.word_ids],
                    batch[output_map.type_ids],
                )
            ):

                if self.config.input_type == "single":
                    cur_offsets_a = (
                        pre_offsets_single[i] if pre_offsets_single else None
                    )
                    cur_offsets_b = None
                else:
                    cur_offsets_a = pre_offsets_a[i] if pre_offsets_a else None
                    cur_offsets_b = pre_offsets_b[i] if pre_offsets_b else None

                fixed = []
                for off, wid, tid in zip(offsets, word_ids, type_ids):
                    fixed.append(
                        self._fix_offset_one(
                            off, wid, tid, cur_offsets_a, cur_offsets_b
                        )
                    )
                fixed_offsets_list.append(fixed)

            batch[output_map.offsets] = fixed_offsets_list

        return batch
