# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Dict, List, Type, Union

import pandas as pd
import torch
from datasets import load_from_disk
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
from dlk.utils.register import register, register_module_name

logger = logging.getLogger(__name__)


@cregister("datamodule", "my_datamodule")
class BasicDatamoduleConfig(Base):
    """the default datamodule"""

    train_batch_size = IntField(
        value=32, minimum=1, help="the batch size of train dataloader"
    )


class MyDataset(Dataset):
    """General Dataset wrapping HF Dataset or Dict"""

    def __init__(
        self,
        data,
        data_type: str,
    ):
        self.data = data
        self.data_type = data_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # Support HF Dataset (dict-like) access
        item = self.data[idx]

        one_ins = {}
        # Convert list to tensor
        one_ins["input_ids"] = torch.tensor(item["input_ids"], dtype=torch.long)
        one_ins["label_ids"] = torch.tensor(item["label_ids"], dtype=torch.long)
        one_ins["_index"] = torch.tensor(idx, dtype=torch.long)
        return one_ins


def data_collate_fn(batch: List[Dict[str, Any]]):
    indexs = [ins["_index"] for ins in batch]
    label_ids = [ins["label_ids"] for ins in batch]
    input_ids = [ins["input_ids"] for ins in batch]

    label_ids = torch.stack(label_ids, dim=0)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    indexs = torch.stack(indexs, dim=0)
    return {"input_ids": input_ids, "label_ids": label_ids, "_index": indexs}


@register("datamodule", "my_datamodule")
class BasicDatamodule(IBaseDataModule):
    """Basic DataModule with setup() for DDP support"""

    def __init__(
        self, config: BasicDatamoduleConfig, data: Dict[str, Any], rt_config: Dict
    ):
        super().__init__()
        self.config = config
        self.rt_config = rt_config
        # 'data' here might contain paths or metadata, actual loading happens in setup
        self.train_data = None
        self.valid_data = None

    def setup(self, stage=None):
        """Load data from disk (memory-mapped) for efficient DDP"""
        processed_dir = (
            "data/processed_data"  # Hardcoded for this example as per processor config
        )

        if stage == "fit" or stage is None:
            train_path = os.path.join(processed_dir, "train")
            valid_path = os.path.join(processed_dir, "valid")

            if os.path.exists(train_path):
                logger.info(f"Loading train data from {train_path}")
                self.train_data = MyDataset(load_from_disk(train_path), "train")

            if os.path.exists(valid_path):
                logger.info(f"Loading valid data from {valid_path}")
                self.valid_data = MyDataset(load_from_disk(valid_path), "valid")

    def train_dataloader(self):
        if not self.train_data:
            return None
        return DataLoader(
            self.train_data,
            batch_size=self.config.train_batch_size,
            collate_fn=data_collate_fn,
            pin_memory=True,
            shuffle=True,
            num_workers=2,  # Safe to use workers with HF datasets
        )

    def val_dataloader(self):
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
        # For online, we wrap the raw dict (list of dicts or dict of lists)
        # Here we assume data is a dict of lists compatible with MyDataset
        return DataLoader(
            MyDataset(data, "online"),
            batch_size=1,
            collate_fn=data_collate_fn,
            shuffle=False,
            num_workers=0,
        )
