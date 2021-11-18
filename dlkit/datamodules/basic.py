import pandas as pd
from torch.utils.data import DataLoader, random_split, Dataset
from typing import Dict, List, Union
from dlkit.utils.config import ConfigTool
from dlkit.datamodules import datamodule_config_register, datamodule_register, IBaseDataModule, collate_register
# from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
import torch
import copy

@datamodule_config_register('basic')
class BasicDatamoduleConfig(object):
    """docstring for BasicDatamoduleConfig
       datamodule: {
           "_name": "basic",
           "config": {
               "pin_memory": None,
               "collate_fn": "default",
               "shuffle": {
                   "train": true,
                   "predict": false,
                   "valid": false,
                   "test": false,
                   "online": false
               },
               "key_type_pairs": {
                    'x': 'float', 
                    'y': 'int'
                },
               "key_padding_pairs": { //default all 0
                    'x': 0, 
                    'y': 0
                },
               "train_batch_size": 32,
               "predict_batch_size": 32, //predict、test batch_size is equals to valid_batch_size
               "online_batch_size": 1,
           }
       }, 
    """
    def __init__(self, config):
        super(BasicDatamoduleConfig, self).__init__()
        config = config.get('config')
        self.key_type_pairs = config.get('key_type_pairs', {})
        self.key_padding_pairs = config.get('key_type_pairs', {})
        self.collate_fn = config.get('collate_fn', 'default')
        self.pin_memory = config.get('pin_memory', False)
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


class BasicDataset(Dataset):
    def __init__(self, key_type_pairs: Dict[str, str], data:pd.DataFrame):
        self.data = data
        self.type_map = {"float": torch.float, "int": torch.long, 'bool': torch.bool, "long": torch.long} 

        self.key_type_pairs = key_type_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        one_ins = {}
        for key, key_type in self.key_type_pairs.items():
            one_ins[key] = torch.tensor(self.data.iloc[idx][key], dtype=self.type_map[key_type])
        one_ins['_index'] = torch.tensor(idx,  dtype=torch.long)
        return one_ins


@datamodule_config_register("basic")
class BasicDatamodule(IBaseDataModule):
    def __init__(self, config: BasicDatamoduleConfig, data: Dict[str, pd.DataFrame]):
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
        self.collate_fn = collate_register.get(config.collate_fn)(key_padding_pairs=config.key_padding_pairs)

    def real_key_type_pairs(self, key_type_pairs: Dict, data: Dict, field: str):
        copy_key_type_pairs = copy.deepcopy(key_type_pairs)
        has_key = set(data[field].columns)
        remove = set()
        for key in copy_key_type_pairs:
            if key not in has_key:
                remove.add(key)
        print(f"There are not '{', '.join(remove)}' in data field {field}.")
        for key in remove:
            copy_key_type_pairs.pop(key)
        return copy_key_type_pairs

    def train_dataloader(self):
        if not self.train_data:
            return None
        return DataLoader(self.train_data, batch_size=self.config.train_batch_size, collate_fn=self.collate_fn, pin_memory=self.config.pin_memory, shuffle=self.config.shuffle.get('train', True))

    def predict_dataloader(self):
        """
        """
        if not self.predict_data:
            return None
        return DataLoader(self.predict_data, batch_size=self.config.predict_batch_size, collate_fn=self.collate_fn, pin_memory=self.config.pin_memory, shuffle=self.config.shuffle.get('predict', False))

    def val_dataloader(self):
        if not self.valid_data:
            return None
        return DataLoader(self.valid_data, batch_size=self.config.valid_batch_size, collate_fn=self.collate_fn, pin_memory=self.config.pin_memory, shuffle=self.config.shuffle.get('valid', False))

    def test_dataloader(self):
        if not self.test_data:
            return None
        return DataLoader(self.test_data, batch_size=self.config.test_batch_size, collate_fn=self.collate_fn, pin_memory=self.config.pin_memory, shuffle=self.config.shuffle.get('test', False))

    def online_dataloader(self):
        # return DataLoader(self.mnist_test, batch_size=self.batch_size)
        return self.collate_fn
