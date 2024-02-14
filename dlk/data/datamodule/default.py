# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Type, Union

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


@cregister("datamodule", "default")
class BasicDatamoduleConfig(Base):
    """the default datamodule"""

    pin_memory = AnyField(
        value=None,
        options=[None, True, False],
        help="whether to use pin_memory, if set to None, we will check the gpu is available",
    )
    num_workers = IntField(
        value=1,
        minimum=-1,
        help="the number of workers for dataloader, if set to -1, we will use os.cpu_count() to get the cpu count",
    )
    shuffle = BoolField(value=True, help="whether shuffle the training data")
    train_batch_size = IntField(
        value=32, minimum=1, help="the batch size of train dataloader"
    )
    predict_batch_size = IntField(
        value=32, minimum=1, help="the batch size of predict dataloader"
    )
    online_batch_size = IntField(
        value=1, minimum=1, help="the batch size of online dataloader"
    )

    submodule = SubModule({}, suggestions=["dataset", "data_collate"])


@register("datamodule", "default")
class BasicDatamodule(IBaseDataModule):
    """Basic and General DataModule"""

    def __init__(
        self, config: BasicDatamoduleConfig, data: Dict[str, Any], rt_config: Dict
    ):
        super().__init__()

        self.config = config
        self.rt_config = rt_config
        self._online_key_type_pairs = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.predict_data = None
        self.dataset_config: DefaultDatasetConfig = config.submodule.dataset
        self.dataset_creator: Type[DefaultDataset] = register.get(
            "dataset", register_module_name(self.dataset_config._module_name)
        )
        if self.config.num_workers == -1:
            import os

            self.config.num_workers = os.cpu_count()
        if "train" in data:
            self.train_data = self.dataset_creator(
                self.dataset_config, data["train"], rt_config
            )
        if "test" in data:
            self.test_data = self.dataset_creator(
                self.dataset_config, data["test"], rt_config
            )
        if "valid" in data:
            self.valid_data = self.dataset_creator(
                self.dataset_config, data["valid"], rt_config
            )
        if "predict" in data:
            self.predict_data = self.dataset_creator(
                self.dataset_config, data["predict"], rt_config
            )
        data_collate_config = config.submodule.data_collate
        self.collate_fn = register.get(
            "data_collate", register_module_name(data_collate_config._module_name)
        )(data_collate_config)

    def train_dataloader(self):
        """get the train set dataloader"""
        if not self.train_data:
            return None
        return DataLoader(
            self.train_data,
            batch_size=self.config.train_batch_size,
            collate_fn=self.collate_fn,
            pin_memory=self.config.pin_memory,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
        )

    def predict_dataloader(self):
        """get the predict set dataloader"""
        if not self.predict_data:
            return None
        return DataLoader(
            self.predict_data,
            batch_size=self.config.predict_batch_size,
            collate_fn=self.collate_fn,
            pin_memory=self.config.pin_memory,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        """get the validation set dataloader"""
        if not self.valid_data:
            return None
        return DataLoader(
            self.valid_data,
            batch_size=self.config.predict_batch_size,
            collate_fn=self.collate_fn,
            pin_memory=self.config.pin_memory,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        """get the test set dataloader"""
        if not self.test_data:
            return None
        return DataLoader(
            self.test_data,
            batch_size=self.config.predict_batch_size,
            collate_fn=self.collate_fn,
            pin_memory=self.config.pin_memory,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def online_dataloader(self, data):
        """get the data collate_fn"""
        # return DataLoader(self.mnist_test, batch_size=self.batch_size)
        if not self._online_key_type_pairs:
            self._online_key_type_pairs = self.dataset_creator.real_key_type_pairs(
                self.dataset_config.key_type_pairs, data
            )
        dataset = self.dataset_creator(
            self.dataset_config, data, self.rt_config, self._online_key_type_pairs
        )
        return DataLoader(
            dataset,
            batch_size=self.config.predict_batch_size,
            collate_fn=self.collate_fn,
            pin_memory=self.config.pin_memory,
            shuffle=False,
            num_workers=1,
        )
