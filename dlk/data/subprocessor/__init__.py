# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import abc
import importlib
import os
from typing import Callable, Dict, Type

import pandas as pd
import pyarrow.parquet as pq
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
    dataclass,
)

from dlk.utils.import_module import import_module_dir


@dataclass
class BaseSubProcessorConfig(Base):
    """the base subprocessor"""

    train_data_set = ListField(
        value=[],
        suggestions=[["train", "valid", "test"]],
        help="the data set should be processed for train stage",
    )
    predict_data_set = ListField(
        value=[],
        suggestions=[["predict"], []],
        help="the data set should be processed for predict stage, only predict data set will be processed or none of the data set will be processed",
    )
    online_data_set = ListField(
        value=[],
        suggestions=[["online"], []],
        help="the data set should be processed for online stage, only online data set will be processed or none of the data set will be processed",
    )
    input_map = DictField(
        value={},
        help="the input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )
    output_map = DictField(
        value={},
        help="the output map of the processor, the key is the name of the processor provided key, the value is the nexted processor needed key",
    )


class BaseSubProcessor(object):
    """docstring for ISubProcessor"""

    def __init__(self, stage: str, config: BaseSubProcessorConfig, meta_dir: str):
        self.loaded_meta = False
        self.meta_dir = meta_dir

    def load_meta(self):
        self.loaded_meta = True

    def process(self, data: pd.DataFrame, deliver_meta: bool) -> pd.DataFrame:
        """SubProcess entry

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
        raise NotImplementedError


# automatically import any Python files in the models directory
subprocessor_dir = os.path.dirname(__file__)
import_module_dir(subprocessor_dir, "dlk.data.subprocessor")
