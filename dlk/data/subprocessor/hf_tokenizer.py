# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from typing import Callable, Dict, Union

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
from tokenizers import normalizers, pre_tokenizers
from transformers import AutoTokenizer

from dlk.utils.io import open
from dlk.utils.register import register
from dlk.utils.tokenizer_util import (
    PreTokenizerFactory,
    TokenizerNormalizerFactory,
    TokenizerPostprocessorFactory,
)

from . import BaseSubProcessor, BaseSubProcessorConfig

logger = logging.getLogger(__name__)


@cregister("subprocessor", "hf_tokenizer")
class HFAutoTokenizerConfig(BaseSubProcessorConfig):
    """HFAutoTokenizer use huggingface transformers.AutoTokenizer"""

    train_data_set = ListField(
        value=["train", "valid", "test"],
        suggestions=[["train", "valid", "test"]],
        help="the data set should be processed for train stage",
    )
    predict_data_set = ListField(
        value=["predict"],
        suggestions=[["predict"], []],
        help="the data set should be processed for predict stage, only predict data set will be processed or none of the data set will be processed",
    )
    online_data_set = ListField(
        value=["online"],
        suggestions=[["online"], []],
        help="the data set should be processed for online stage, only online data set will be processed or none of the data set will be processed",
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
            value="pretokenized_words", help="pretokenized word related to sentence"
        )
        pretokenized_words_a = StrField(
            value="pretokenized_words_a",
            help="pretokenized word b related to sentence_a",
        )
        pretokenized_words_b = StrField(
            value="pretokenized_words_b",
            help="pretokenized word b related to sentence_b",
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

    input_map = NestField(
        value=InputMap,
        help="the input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )

    class OutputMap:
        tokens = StrField(value="", help="the output tokens")
        ids = StrField(value="input_ids", help="the output input_ids")
        attention_mask = StrField(
            value="attention_mask", help="the output attention_mask"
        )
        type_ids = StrField(value="token_type_ids", help="the output token_type_ids")
        special_tokens_mask = StrField(value="", help="the output special_tokens_mask")
        overflowing = StrField(value="", help="the output overflowing_tokens")
        offsets = StrField(value="offset_mapping", help="the output offset_mapping")
        word_ids = StrField(value="word_ids", help="the output word_ids")
        sequence_ids = StrField(value="sequence_ids", help="the output sequence_ids")

    output_map = NestField(
        value=OutputMap,
        help="the output map of the processor, the key is the name of the processor provided key, the value is the nexted processor needed key",
    )
    tokenizer_path = StrField(
        value=MISSING,
        help="the config path for the tokenizer, this is not effected by `meta_dir`",
    )

    class Truncation:
        direction = StrField(
            value="right",
            options=["right", "left"],
            help="the truncation direction",
        )
        stride = IntField(value=0, help="the stride for truncation")
        max_length = IntField(
            value=512, minimum=1, help="the max length for truncation"
        )
        strategy = StrField(
            value="longest_first",
            options=["longest_first", "only_first", "only_second"],
            help="the truncation strategy",
        )

    truncation = NestField(value=Truncation)
    normalizer = AnyField(
        value="default",
        suggestions=[
            "default",
            [
                "nfd",
                "lowercase",
                "strip_accents",
                {"some_processor_need_config": {}},
            ],
        ],
        help="the normalizer for the tokenizer, list of normalizer",
    )
    pre_tokenizer = AnyField(
        value="default",
        suggestions=[
            "default",
            [
                "whitespace",
                "whitespacesplit",
                "bytelevel",
                "bert",
                {"some_processor_need_config": {}},
            ],
        ],
        help="the pre tokenizer for the tokenizer, if not default, you can provide a list of pre tokenizers",
    )

    class ProcessData:
        is_pretokenized = BoolField(
            value=False, help="whether the input is pretokenized"
        )
        add_special_tokens = BoolField(value=True, help="whether add special tokens")

    process_data = NestField(value=ProcessData)
    input_type = StrField(
        value="single",
        options=["single", "pair"],
        help="the input type of the tokenizer, single or pair",
    )
    fix_offset = BoolField(
        value=False, help="whether fix the offset for the pretokenized word"
    )


@register("subprocessor", "hf_tokenizer")
class HFAutoTokenizer(BaseSubProcessor):
    """HFAutoTokenizer use hugingface transformers.AutoTokenizer

    Tokenize the single $sentence
    Or tokenizer the pair $sentence_a, $sentence_b
    Generator $tokens, $input_ids, $token_type_ids, $special_tokens_mask, $offset_mapping, $word_ids, $overflowing_tokens, $sequence_ids
    """

    def __init__(self, stage: str, config: HFAutoTokenizerConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path, use_fast=True
        )

        # The `backend_tokenizer` attribute gives us access to the underlying `tokenizers` library object
        backend_tokenizer = self.tokenizer.backend_tokenizer
        pretokenizer_factory = PreTokenizerFactory(backend_tokenizer)
        tokenizer_normalizer_factory = TokenizerNormalizerFactory(backend_tokenizer)

        if not self.config.pre_tokenizer:
            backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])
        elif self.config.pre_tokenizer != "default":
            assert isinstance(self.config.pre_tokenizer, list)
            pre_tokenizers_list = [
                self._get_processor(pretokenizer_factory, one_pre_tokenizer)
                for one_pre_tokenizer in self.config.pre_tokenizer
            ]
            backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                pre_tokenizers_list
            )

        if not self.config.normalizer:
            backend_tokenizer.normalizer = normalizers.Sequence([])
        elif self.config.normalizer != "default":
            assert isinstance(self.config.normalizer, list)
            normalizers_list = [
                self._get_processor(tokenizer_normalizer_factory, one_normalizer)
                for one_normalizer in self.config.normalizer
            ]
            backend_tokenizer.normalizer = normalizers.Sequence(normalizers_list)

    def _get_processor(
        self,
        factory: Union[
            PreTokenizerFactory,
            TokenizerNormalizerFactory,
        ],
        one_processor: Union[Dict, str],
    ):
        """return the processor in factory by the processor name and update the config of the processor if provide

        Args:
            factory: process factory
            one_processor: the processor info, it's name (and config)

        Returns:
            processor

        """
        if isinstance(one_processor, dict):
            assert len(one_processor) == 1
            process_name, process_config = list(one_processor.items())[0]
            return factory.get(process_name)(**process_config)
        else:
            assert isinstance(one_processor, str)
            return factory.get(one_processor)()

    def process(self, data: Dict) -> Dict:
        """haggingface tokenizer entry

        Args:
            data: Dict or Dict Like
            >>> {"sentence": ["sent_a", "sent_b"], "label": ["la", "lb"]}

        Returns:
            updated data
        """
        truncation_strategy = (
            self.config.truncation.strategy
            if self.config.truncation
            else "do_not_truncate"
        )

        if self.config.input_type == "single":
            sentences = (
                data[self.config.input_map.pretokenized_words]
                if self.config.process_data.is_pretokenized
                else data[self.config.input_map.sentence]
            )
            batch_encodes = self.tokenizer(
                sentences,
                is_split_into_words=self.config.process_data.is_pretokenized,
                add_special_tokens=self.config.process_data.add_special_tokens,
                truncation=truncation_strategy,
                max_length=self.config.truncation.max_length
                if self.config.truncation
                else None,
                stride=self.config.truncation.stride if self.config.truncation else 0,
                return_overflowing_tokens=self.config.output_map.overflowing != "",
                return_offsets_mapping=True,
                padding="max_length" if self.config.truncation else False,
            )
        else:  # pair
            sentence_as = (
                data[self.config.input_map.pretokenized_words_a]
                if self.config.process_data.is_pretokenized
                else data[self.config.input_map.sentence_a]
            )
            sentence_bs = (
                data[self.config.input_map.pretokenized_words_b]
                if self.config.process_data.is_pretokenized
                else data[self.config.input_map.sentence_b]
            )
            batch_encodes = self.tokenizer(
                sentence_as,
                sentence_bs,
                is_split_into_words=self.config.process_data.is_pretokenized,
                add_special_tokens=self.config.process_data.add_special_tokens,
                truncation=truncation_strategy,
                max_length=self.config.truncation.max_length
                if self.config.truncation
                else None,
                stride=self.config.truncation.stride if self.config.truncation else 0,
                return_overflowing_tokens=self.config.output_map.overflowing != "",
                return_offsets_mapping=True,
                padding="max_length" if self.config.truncation else False,
            )

        output_map = self.config.output_map
        if output_map.ids:
            data[output_map.ids] = batch_encodes["input_ids"]
        if output_map.attention_mask:
            data[output_map.attention_mask] = batch_encodes["attention_mask"]
        if output_map.type_ids:
            data[output_map.type_ids] = batch_encodes["token_type_ids"]
        if output_map.offsets:
            data[output_map.offsets] = batch_encodes["offset_mapping"]
        if output_map.special_tokens_mask and "special_tokens_mask" in batch_encodes:
            data[output_map.special_tokens_mask] = batch_encodes["special_tokens_mask"]
        if output_map.overflowing:
            data[output_map.overflowing] = batch_encodes["overflowing_tokens"]

        if output_map.tokens:
            tokens_list = [
                self.tokenizer.convert_ids_to_tokens(ids)
                for ids in batch_encodes["input_ids"]
            ]
            data[output_map.tokens] = tokens_list
        if output_map.word_ids:
            word_ids_list = [
                batch_encodes.word_ids(i)
                for i in range(len(batch_encodes["input_ids"]))
            ]
            data[output_map.word_ids] = word_ids_list
        if output_map.sequence_ids:
            sequence_ids_list = [
                batch_encodes.sequence_ids(i)
                for i in range(len(batch_encodes["input_ids"]))
            ]
            data[output_map.sequence_ids] = sequence_ids_list

        if self.config.process_data.is_pretokenized and self.config.fix_offset:
            data[output_map.offsets] = self._fix_offset(data)
        return data

    def _fix_offset(self, data):
        """fix the pretokenizerd offset

        Args:
            data: the dataset

        Returns:
            fixed offsets

        """
        word_offset_a_list = data[
            self.config.input_map.pretokenized_word_offsets_a
            if self.config.input_type == "pair"
            else self.config.input_map.pretokenized_word_offsets
        ]
        word_offset_b_list = data[
            self.config.input_map.pretokenized_word_offsets_b
            if self.config.input_type == "pair"
            else self.config.input_map.pretokenized_word_offsets
        ]
        offsets_list = data[self.config.output_map.offsets]
        word_ids_list = data[self.config.output_map.word_ids]
        type_ids_list = data[self.config.output_map.type_ids]

        fixed_offsets_list = []
        for word_offset_a, word_offset_b, offsets, word_ids, type_ids in zip(
            word_offset_a_list,
            word_offset_b_list,
            offsets_list,
            word_ids_list,
            type_ids_list,
        ):
            fixed_offsets = []
            word_offsets = [word_offset_a, word_offset_b]
            for offset, word_id, type_id in zip(offsets, word_ids, type_ids):
                if offset == (0, 0) or word_id is None:
                    fixed_offsets.append(offset)
                else:
                    fixed_offsets.append(
                        (
                            offset[0] + word_offsets[type_id][word_id][0],
                            offset[1] + word_offsets[type_id][word_id][0],
                        )
                    )
            fixed_offsets_list.append(fixed_offsets)
        return fixed_offsets_list
