# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Type, Union

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
from torch.utils.data import DataLoader, Dataset

from dlk.data.datamodule import IBaseDataModule
from dlk.data.dataset.default import DefaultDataset, DefaultDatasetConfig
from dlk.utils.register import register, register_module_name

logger = logging.getLogger(__name__)


@cregister("datamodule", "my_datamodule")
class BasicDatamoduleConfig(Base):
    """the default datamodule"""

    train_batch_size = IntField(
        value=32, minimum=1, help="the batch size of train dataloader"
    )


class MyDataset(Dataset):
    """General Dataset"""

    def __init__(
        self,
        data: pd.DataFrame,
        data_type: str,
    ):
        """
        if repeat_valid >1 we will repeat each item in this dataset
        """
        self.data = data
        self.data_type = data_type

    def __len__(self):
        """return the dataset size"""
        return len(self.data)

    def __getitem__(self, idx: int):
        """return one instance by index

        Args:
            idx: the index of data

        Returns:
            the data[idx] and convert to tensor the result will add 'idx' to '_index'

        """
        one_ins = {}
        one_ins["input_ids"] = torch.tensor(
            self.data.iloc[idx]["input_ids"], dtype=torch.long
        )
        one_ins["label_ids"] = torch.tensor(
            self.data.iloc[idx]["label_ids"], dtype=torch.long
        )
        one_ins["_index"] = torch.tensor(idx, dtype=torch.long)
        return one_ins


def data_collate_fn(batch: List[Dict[str, Any]]):
    """concat and padding the batch data, the simple version, you should add gen mask as the dlk default collection function do

    Args:
        batch: batch data

    Returns:

    """
    indexs = [ins["_index"] for ins in batch]
    label_ids = [ins["label_ids"] for ins in batch]
    input_ids = [ins["input_ids"] for ins in batch]

    label_ids = torch.stack(label_ids, dim=0)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    indexs = torch.stack(indexs, dim=0)
    return {"input_ids": input_ids, "label_ids": label_ids, "_index": indexs}


@register("datamodule", "my_datamodule")
class BasicDatamodule(IBaseDataModule):
    """Basic and General DataModule"""

    def __init__(
        self, config: BasicDatamoduleConfig, data: Dict[str, Any], rt_config: Dict
    ):
        super().__init__()

        self.config = config
        self.train_data = MyDataset(data["train"], "train") if "train" in data else None
        self.valid_data = MyDataset(data["valid"], "valid") if "valid" in data else None

    def train_dataloader(self):
        """get the train set dataloader"""
        if not self.train_data:
            return None
        return DataLoader(
            self.train_data,
            batch_size=self.config.train_batch_size,
            collate_fn=data_collate_fn,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        """get the valid set dataloader"""
        if not self.valid_data:
            return None
        return DataLoader(
            self.valid_data,
            batch_size=1,
            collate_fn=data_collate_fn,
            pin_memory=True,
            shuffle=False,
        )

    def online_dataloader(self, data):
        """get the online stage dataloader"""
        return DataLoader(
            MyDataset(data, "online"),
            batch_size=1,
            collate_fn=data_collate_fn,
            shuffle=False,
            num_workers=1,
        )
