# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from typing import Callable, Dict, Iterable, List, Set, Union

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

from dlk.utils.io import open
from dlk.utils.register import register
from dlk.utils.vocab import Vocabulary

from . import BaseSubProcessor, BaseSubProcessorConfig

logger = logging.getLogger(__name__)


@cregister("subprocessor", "token_gather")
class TokenGatherConfig(BaseSubProcessorConfig):
    """the token gather subprocessor"""

    train_data_set = ListField(
        value=["train"],
        suggestions=[["train", "valid", "test"]],
        help="the data set should be processed for train stage",
    )
    gather_columns = ListField(
        value=MISSING,
        suggestions=[
            ["tokens"],
            [
                "tokens",
                {"column": "entities_column", "trace": "entities_info.labels"},
            ],
        ],
        help="List of columns. If the column is a reprent as a dict,  we will trace the real elements by 'trace'. for example: {'entities_info': [{'start': 1， 'end': 2, labels: ['Label1']}, ..]}, the trace to labels is 'entities_info.labels'",
    )
    token_vocab = StrField(
        value="token_vocab.json",
        help="save Vocabulary object (the Vocabulary of labels) file.",
    )
    ignore = StrField(
        value="",
        help="ignore the token, the id of this token will be -100, null means nothing should be ignore",
    )
    update = StrField(
        value="",
        help="null or another exists Vocabulary object should be update, if the update is not null, the unk and pad ignore will be ignored",
    )
    unk = StrField(value="[UNK]", additions=[None], help="the unk token")
    pad = StrField(value="[PAD]", additions=[None], help="the pad token")
    min_freq = IntField(
        value=1,
        minimum=1,
        help="the min freq of token, you can only change one of the value of min_freq and most_common",
    )
    most_common = IntField(
        value=-1,
        minimum=-1,
        help="the most common token, -1 for all, you can only change one of the value of min_freq and most_common.",
    )


@register("subprocessor", "token_gather")
class TokenGather(BaseSubProcessor):
    """gather all tokens from the 'gather_columns' and deliver a vocab named 'token_vocab'"""

    def __init__(self, stage: str, config: TokenGatherConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.stage = stage
        self.config = config
        self.update = (
            None
            if not self.config.update
            else os.path.join(self.meta_dir, self.config.update)
        )

    def get_elements_from_series_by_trace(self, data: pd.Series, trace: str) -> List:
        """get the data from data[trace_path]
        >>> for example:
        >>> data[0] = {'entities_info': [{'start': 0, 'end': 1, 'labels': ['Label1']}]} // data is a series, and every element is as data[0]
        >>> trace = 'entities_info.labels'
        >>> return_result = [['Label1']]

        Args:
            data: origin data series
            trace: get data element trace

        Returns:
            the data in the tail of traces

        """

        def get_elements_from_iter_by_trace(iter: Iterable, cur_trace_list: List):
            if not cur_trace_list:
                return iter
            if isinstance(iter, dict):
                return get_elements_from_iter_by_trace(
                    iter[cur_trace_list[0]], cur_trace_list[1:]
                )
            if isinstance(iter, list) or isinstance(iter, tuple):
                return [
                    get_elements_from_iter_by_trace(sub_iter, cur_trace_list)
                    for sub_iter in iter
                ]
            raise PermissionError(
                f"The trace path is only support type list and dict, but you provide {type(iter)}"
            )

        return [get_elements_from_iter_by_trace(one, trace.split(".")) for one in data]

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
        if self.update:
            with open(self.update, mode="r", encoding="utf-8") as f:
                self.vocab = Vocabulary.load(json.load(f))
        else:
            self.vocab = Vocabulary(
                do_strip=True, unknown=self.config.unk, ignore=self.config.ignore
            )
        for column in self.config.gather_columns:
            if isinstance(column, str):
                self.vocab.auto_update(data[column])
            elif isinstance(column, dict):
                self.vocab.auto_update(
                    self.get_elements_from_series_by_trace(
                        data[column["column"]], trace=column["trace"]
                    )
                )
            else:
                raise PermissionError(
                    f"The gather column currently is only support str or dict."
                )
        self.vocab.filter_rare(self.config.min_freq, self.config.most_common)
        logger.info(f"The Vocab Num is {self.vocab.word_num}")
        with open(
            os.path.join(self.meta_dir, self.config.token_vocab), "w", encoding="utf-8"
        ) as f:
            json.dump(self.vocab.dumps(), f)

        return data
