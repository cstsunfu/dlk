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
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordPiece

from dlk.utils.io import open
from dlk.utils.register import register
from dlk.utils.tokenizer_util import (
    PreTokenizerFactory,
    TokenizerNormalizerFactory,
    TokenizerPostprocessorFactory,
)

from . import BaseSubProcessor, BaseSubProcessorConfig

logger = logging.getLogger(__name__)


@cregister("subprocessor", "fast_tokenizer")
class FastTokenizerConfig(BaseSubProcessorConfig):
    """FastTokenizer use hugingface tokenizers"""

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
        tokens = StrField(value="tokens", help="the output tokens")
        ids = StrField(value="input_ids", help="the output input_ids")
        attention_mask = StrField(
            value="attention_mask", help="the output attention_mask"
        )
        type_ids = StrField(value="type_ids", help="the output type_ids")
        special_tokens_mask = StrField(
            value="special_tokens_mask", help="the output special_tokens_mask"
        )
        overflowing = StrField(value="overflowing", help="the output overflowing")
        offsets = StrField(value="offsets", help="the output offsets")
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
    post_processor = AnyField(
        value="default",
        suggestions=["default", "bert", {"some_processor_need_config": {}}],
        help="the post processor for the tokenizer",
    )

    class ProcessData:
        is_pretokenized = BoolField(
            value=False, help="whether the input is pretokenized"
        )
        add_special_tokens = BoolField(value=True, help="whether add special tokens")

    process_data = NestField(value=ProcessData)
    expand_examples = BoolField(
        value=False,
        help="if the sequence is very long will split to multiple instance, whether expand the examples",
    )
    input_type = StrField(
        value="single",
        options=["single", "pair"],
        help="the input type of the tokenizer, single or pair",
    )
    fix_offset = BoolField(
        value=False, help="whether fix the offset for the pretokenized word"
    )


@register("subprocessor", "fast_tokenizer")
class FastTokenizer(BaseSubProcessor):
    """FastTokenizer use hugingface tokenizers

    Tokenizer the single $sentence
    Or tokenizer the pair $sentence_a, $sentence_b
    Generator $tokens, $input_ids, $type_ids, $special_tokens_mask, $offsets, $word_ids, $overflowing, $sequence_ids
    """

    def __init__(self, stage: str, config: FastTokenizerConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.config = config
        with open(self.config.tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_str = json.dumps(json.load(f))

        self.tokenizer = Tokenizer.from_str(tokenizer_str)
        pretokenizer_factory = PreTokenizerFactory(self.tokenizer)
        tokenizer_postprocessor_factory = TokenizerPostprocessorFactory(self.tokenizer)
        tokenizer_normalizer_factory = TokenizerNormalizerFactory(self.tokenizer)

        if not self.config.pre_tokenizer:
            self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])
        elif self.config.pre_tokenizer != "default":
            assert isinstance(self.config.pre_tokenizer, list)
            pre_tokenizers_list = []
            for one_pre_tokenizer in self.config.pre_tokenizer:
                pre_tokenizers_list.append(
                    self._get_processor(pretokenizer_factory, one_pre_tokenizer)
                )
            self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pre_tokenizers_list)

        if not self.config.post_processor:
            raise KeyError(
                "The tokenizer is not support disable default tokenizers post processor. (You can delete the config manually)"
            )
        elif self.config.post_processor != "default":
            self.tokenizer.post_processor = self._get_processor(
                tokenizer_postprocessor_factory, self.config.post_processor
            )

        if not self.config.normalizer:
            self.tokenizer.normalizer = normalizers.Sequence([])
        elif self.config.normalizer != "default":
            assert isinstance(self.config.normalizer, list)
            normalizers_list = []
            for one_normalizer in self.config.normalizer:
                normalizers_list.append(
                    self._get_processor(tokenizer_normalizer_factory, one_normalizer)
                )
            self.tokenizer.normalizer = normalizers.Sequence(normalizers_list)

        if self.config.truncation:
            self.tokenizer.enable_truncation(
                max_length=self.config.truncation.max_length,
                stride=self.config.truncation.stride,
                strategy=self.config.truncation.strategy,
                direction=self.config.truncation.direction,
            )

    def _get_processor(
        self,
        factory: Union[
            PreTokenizerFactory,
            TokenizerNormalizerFactory,
            TokenizerPostprocessorFactory,
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

    def process(self, data: pd.DataFrame, deliver_meta: bool) -> pd.DataFrame:
        """use self._tokenize tokenize the data

        Args:
            data: several data in dataframe

        Returns:
            updated dataframe

        """
        if self.config.input_type == "single":
            batch_encodes = self.tokenizer.encode_batch(
                data[self.config.input_map.pretokenized_words]
                if self.config.process_data.is_pretokenized
                else data[self.config.input_map.sentence],
                is_pretokenized=self.config.process_data.is_pretokenized,
                add_special_tokens=self.config.process_data.add_special_tokens,
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
            batch_encodes = self.tokenizer.encode_batch(
                [
                    (sentence_a, sentence_b)
                    for sentence_a, sentence_b in zip(sentence_as, sentence_bs)
                ],
                is_pretokenized=self.config.process_data.is_pretokenized,
                add_special_tokens=self.config.process_data.add_special_tokens,
            )
        if self.config.expand_examples:
            encodes_list = []
            for encode in batch_encodes:
                encodes_list.append([encode] + encode.overflowing)
            data["_tokenizer_encoders"] = encodes_list
            data = data.explode("_tokenizer_encoders", ignore_index=True)
        else:
            data["_tokenizer_encoders"] = batch_encodes
        (
            tokens_list,
            ids_list,
            attention_mask_list,
            type_ids_list,
            special_tokens_mask_list,
            offsets_list,
            word_ids_list,
            overflowing_list,
            sequence_ids_list,
        ) = ([], [], [], [], [], [], [], [], [])
        for encode in data["_tokenizer_encoders"]:
            tokens_list.append(encode.tokens)
            ids_list.append(encode.ids)
            attention_mask_list.append(encode.attention_mask)
            type_ids_list.append(encode.type_ids)
            special_tokens_mask_list.append(encode.special_tokens_mask)
            offsets_list.append(encode.offsets)
            word_ids_list.append(encode.word_ids)
            # overflowing_list.append(encode.overflowing)
            sequence_ids_list.append(encode.sequence_ids)
        output_map = self.config.output_map
        data[output_map.tokens] = tokens_list
        data[output_map.ids] = ids_list
        data[output_map.attention_mask] = attention_mask_list
        data[output_map.type_ids] = type_ids_list
        data[output_map.special_tokens_mask] = special_tokens_mask_list
        data[output_map.offsets] = offsets_list
        data[output_map.word_ids] = word_ids_list
        # data[output_map['overflowing']] = overflowing_list
        data[output_map.sequence_ids] = sequence_ids_list
        data.drop("_tokenizer_encoders", axis=1, inplace=True)

        if self.config.process_data.is_pretokenized and self.config.fix_offset:
            data[output_map.offsets] = data.apply(self._fix_offset, axis=1)
        return data

    def _fix_offset(self, one_line: pd.Series):
        """fix the pretokenizerd offset

        Args:
            one_line: a Series which contains the config.input_map.pretokenized_word_offsets, config.output_map.offsets, config.output_map.word_ids, configs.output_map.type_ids

        Returns:
            encode.tokens, encode.ids, encode.attention_mask, encode.type_ids, encode.special_tokens_mask, encode.offsets, encode.word_ids, encode.overflowing, encode.sequence_ids

        """
        fixed_offsets = []
        word_offsets = []
        if self.config.input_type == "single":
            word_offsets = [one_line[self.config.input_map.pretokenized_word_offsets]]
        else:  # pair
            word_offsets = [
                one_line[self.config.input_map.pretokenized_word_offsets_a],
                one_line[self.config.input_map.pretokenized_word_offsets_b],
            ]
        for offset, word_id, type_id in zip(
            one_line["offsets"], one_line["word_ids"], one_line["type_ids"]
        ):
            if offset == (0, 0):
                fixed_offsets.append(offset)
            else:
                fixed_offsets.append(
                    (
                        offset[0] + word_offsets[type_id][word_id][0],
                        offset[1] + word_offsets[type_id][word_id][0],
                    )
                )
        return fixed_offsets
