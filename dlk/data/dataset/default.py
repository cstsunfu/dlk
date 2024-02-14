# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from typing import Any, Dict

import pandas as pd
import torch
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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from dlk.utils.register import register


@cregister("dataset", "default")
class DefaultDatasetConfig(Base):
    """the default dataset"""

    key_type_pairs = DictField(value={}, help="the pair of key and type")
    repeat_for_valid = BoolField(
        value=True, help="whether to repeat the data for valid"
    )


@register("dataset", "default")
class DefaultDataset(Dataset):
    """General Dataset"""

    def __init__(
        self,
        config: DefaultDatasetConfig,
        data: pd.DataFrame,
        rt_config: Dict,
        key_type_pairs: Dict = None,
    ):
        """
        if repeat_valid >1 we will repeat each item in this dataset
        """
        self.config = config

        self.repeat_valid = 1
        if self.config.repeat_for_valid and rt_config.get("world_size", 1) > 1:
            self.repeat_valid = rt_config.get("world_size", 1)
        self.data = data
        self.type_map = {
            "float": torch.float,
            "int": torch.int,
            "bool": torch.bool,
            "long": torch.long,
        }
        if key_type_pairs is not None:
            self.key_type_pairs = key_type_pairs
        else:
            self.key_type_pairs = self.real_key_type_pairs(config.key_type_pairs, data)

    @staticmethod
    def real_key_type_pairs(key_type_pairs: Dict, data: pd.DataFrame):
        """return the keys = key_type_pairs.keys() ∩ data.columns

        Args:
            key_type_pairs: data in columns should map to tensor type
            data: the pd.DataFrame

        Returns:
            real_key_type_pairs where keys = key_type_pairs.keys() ∩ data.columns

        """
        copy_key_type_pairs = copy.deepcopy(key_type_pairs)
        has_key = set(data.columns)
        remove = set()
        for key in copy_key_type_pairs:
            if key not in has_key:
                remove.add(key)
        for key in remove:
            copy_key_type_pairs.pop(key)
        return copy_key_type_pairs

    def __len__(self):
        """return the dataset size"""
        return len(self.data) * self.repeat_valid

    def __getitem__(self, idx: int):
        """return one instance by index

        Args:
            idx: the index of data

        Returns:
            the data[idx] and convert to tensor the result will add 'idx' to '_index'

        """
        idx = idx // self.repeat_valid
        one_ins = {}
        for key, key_type in self.key_type_pairs.items():
            one_ins[key] = torch.tensor(
                self.data.iloc[idx][key], dtype=self.type_map[key_type]
            )
        one_ins["_index"] = torch.tensor(idx, dtype=torch.long)
        return one_ins
