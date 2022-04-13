# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
from torch.utils.data import DataLoader, random_split, Dataset
from typing import Dict, List, Union, Any
from dlk.utils.config import BaseConfig, ConfigTool
from dlk.data.datamodules import datamodule_config_register, datamodule_register, IBaseDataModule, collate_register
from dlk.utils.logger import Logger
# from pytorch_lightning import LightningDataModule
import os
from torch.nn.utils.rnn import pad_sequence
import torch
import copy
logger = Logger.get_logger()

@datamodule_config_register('basic')
class BasicDatamoduleConfig(BaseConfig):
    """Config for BasicDatamodule

    Config Example:
        >>> {
        >>>     "_name": "basic",
        >>>     "config": {
        >>>         "pin_memory": None,
        >>>         "collate_fn": "default",
        >>>         "num_workers": null,
        >>>         "shuffle": {
        >>>             "train": true,
        >>>             "predict": false,
        >>>             "valid": false,
        >>>             "test": false,
        >>>             "online": false
        >>>         },
        >>>         "key_type_pairs": {
        >>>              'input_ids': 'int',
        >>>              'label_ids': 'long',
        >>>              'type_ids': 'long',
        >>>          },
        >>>         "gen_mask": {
        >>>              'input_ids': 'attention_mask',
        >>>          },
        >>>         "key_padding_pairs": { //default all 0
        >>>              'input_ids': 0,
        >>>          },
        >>>         "key_padding_pairs_2d": { //default all 0, for 2 dimension data
        >>>              'input_ids': 0,
        >>>          },
        >>>         "train_batch_size": 32,
        >>>         "predict_batch_size": 32, //predict、test batch_size is equals to valid_batch_size
        >>>         "online_batch_size": 1,
        >>>     }
        >>> },
    """
    def __init__(self, config):
        super(BasicDatamoduleConfig, self).__init__(config)
        config = config['config']
        self.key_type_pairs = config.get('key_type_pairs', {})

        if "_index" in self.key_type_pairs:
            logger.warning(f"'_index' is a preserved key, we will use this to indicate the index of item, if you ignore this warning, we will ignore the origin '_index' data.")
        self.key_padding_pairs = config.get('key_padding_pairs', {})
        self.key_padding_pairs_2d = config.get('key_padding_pairs_2d', {})
        self.gen_mask = config.get("gen_mask", {})
        self.collate_fn = config.get('collate_fn', 'default')
        self.pin_memory = config.get('pin_memory', False)
        self.num_workers = config.get('num_workers', 0) if config.get('num_workers', 0) else os.cpu_count()
        if self.pin_memory is None:
            self.pin_memory = torch.cuda.is_available()
        self.shuffle = config.get('shuffle', {
            "train": True,
            "predict": False,
            "valid": False,
            "test": False,
            "online": False
        })
        self.train_batch_size = config.get('train_batch_size', 32)
        self.test_batch_size = config.get('predict_batch_size', 32)
        self.valid_batch_size = config.get('predict_batch_size', 32)
        self.predict_batch_size = config.get('predict_batch_size', 32)
        self.online_batch_size = config.get('online_batch_size', 1)
        self.post_check(config, used=[
               "pin_memory",
               "collate_fn",
               "num_workers",
               "shuffle",
               "key_type_pairs",
               "gen_mask",
               "key_padding_pairs",
               "key_padding_pairs_2d",
               "train_batch_size",
               "predict_batch_size",
               "online_batch_size",
        ])


class BasicDataset(Dataset):
    """Basic and General Dataset"""
    def __init__(self, key_type_pairs: Dict[str, str], data:pd.DataFrame):
        self.data = data
        self.type_map = {"float": torch.float, "int": torch.int, 'bool': torch.bool, "long": torch.long}

        self.key_type_pairs = key_type_pairs

    def __len__(self):
        """return teh dataset size
        """
        return len(self.data)

    def __getitem__(self, idx: int):
        """return one instance by index

        Args:
            idx: the index of data

        Returns: 
            the data[idx] and convert to tensor the result will add 'idx' to '_index'

        """
        one_ins = {}
        for key, key_type in self.key_type_pairs.items():
            one_ins[key] = torch.tensor(self.data.iloc[idx][key], dtype=self.type_map[key_type])
        one_ins['_index'] = torch.tensor(idx,  dtype=torch.long)
        return one_ins


@datamodule_register("basic")
class BasicDatamodule(IBaseDataModule):
    """Basic and General DataModule
    """
    def __init__(self, config: BasicDatamoduleConfig, data: Dict[str, Any]):
        super().__init__()

        self.config = config
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.predict_data = None
        if 'train' in data:
            self.train_data = BasicDataset(self.real_key_type_pairs(config.key_type_pairs, data, 'train'), data['train'])
        if "test" in data:
            self.test_data = BasicDataset(self.real_key_type_pairs(config.key_type_pairs, data, 'test'), data['test'])
        if "valid" in data:
            self.valid_data = BasicDataset(self.real_key_type_pairs(config.key_type_pairs, data, 'valid'), data['valid'])
        if "predict" in data:
            self.predict_data = BasicDataset(self.real_key_type_pairs(config.key_type_pairs, data, 'predict'), data['predict'])
        self.collate_fn = collate_register.get(config.collate_fn)(key_padding_pairs=config.key_padding_pairs, gen_mask=config.gen_mask, key_padding_pairs_2d=config.key_padding_pairs_2d)

    def real_key_type_pairs(self, key_type_pairs: Dict, data: Dict, field: str):
        """return the keys = key_type_pairs.keys() ∩ data.columns

        Args:
            key_type_pairs: data in columns should map to tensor type
            data: the pd.DataFrame
            field: traing/valid/test, etc.

        Returns: 
            real_key_type_pairs where keys = key_type_pairs.keys() ∩ data.columns

        """
        copy_key_type_pairs = copy.deepcopy(key_type_pairs)
        has_key = set(data[field].columns)
        remove = set()
        for key in copy_key_type_pairs:
            if key not in has_key:
                remove.add(key)
        if remove:
            logger.warning(f"There are not '{', '.join(remove)}' in data field {field}.")
        for key in remove:
            copy_key_type_pairs.pop(key)
        return copy_key_type_pairs

    def train_dataloader(self):
        """get the train set dataloader"""
        if not self.train_data:
            return None
        return DataLoader(self.train_data, batch_size=self.config.train_batch_size, collate_fn=self.collate_fn, pin_memory=self.config.pin_memory, shuffle=self.config.shuffle.get('train', True), num_workers=self.config.num_workers)

    def predict_dataloader(self):
        """get the predict set dataloader"""
        if not self.predict_data:
            return None
        return DataLoader(self.predict_data, batch_size=self.config.predict_batch_size, collate_fn=self.collate_fn, pin_memory=self.config.pin_memory, shuffle=self.config.shuffle.get('predict', False), num_workers=self.config.num_workers)

    def val_dataloader(self):
        """get the validation set dataloader"""
        if not self.valid_data:
            return None
        return DataLoader(self.valid_data, batch_size=self.config.valid_batch_size, collate_fn=self.collate_fn, pin_memory=self.config.pin_memory, shuffle=self.config.shuffle.get('valid', False), num_workers=self.config.num_workers)

    def test_dataloader(self):
        """get the test set dataloader"""
        if not self.test_data:
            return None
        return DataLoader(self.test_data, batch_size=self.config.test_batch_size, collate_fn=self.collate_fn, pin_memory=self.config.pin_memory, shuffle=self.config.shuffle.get('test', False), num_workers=self.config.num_workers)

    def online_dataloader(self):
        """get the data collate_fn"""
        # return DataLoader(self.mnist_test, batch_size=self.batch_size)
        return self.collate_fn
