import pandas as pd
from torch.utils.data import DataLoader, random_split, Dataset
from typing import Dict, List, Union
from dlkit.utils.config import Config
from dlkit.datamodules import datamodule_config_register, datamodule_register, IBaseDataModule
# from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
import torch

@datamodule_config_register('basic')
class BasicDatamoduleConfig(Config):
    """docstring for BasicDatamoduleConfig
       datamodule: {
           "_name": "basic",
           "config": {
               "padding": 0,
               "pin_memory": None,
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
               "train_batch_size": 32,
               "predict_batch_size": 32, //predict、test batch_size is equals to valid_batch_size
               "online_batch_size": 1,
           }
       }, 
    """
    def __init__(self, **kwargs):
        super(BasicDatamoduleConfig, self).__init__(**kwargs)
        self.key_type_pairs = kwargs.get('key_type_pairs', {})
        self.pin_memory = kwargs.get('pin_memory', False)
        if self.pin_memory is None:
            self.pin_memory = torch.cuda.is_available()
        self.shuffle = kwargs.get('shuffle', { 
            "train": True, 
            "predict": False, 
            "valid": False,
            "test": False,
            "online": False 
        })
        self.train_batch_size = kwargs.get('train_batch_size', 32)
        self.test_batch_size = kwargs.get('predict_batch_size', 32)
        self.valid_batch_size = kwargs.get('predict_batch_size', 32)
        self.predict_batch_size = kwargs.get('predict_batch_size', 32)
        self.online_batch_size = kwargs.get('online_batch_size', 1)


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
        one_ins['_index'] = torch.tensor([idx],  dtype=torch.long)
        return one_ins
        

def base_collate(batch):
    keys = batch[0].keys()
    data_map: Dict[str, Union[List[torch.Tensor], torch.Tensor]] = {}
    for key in keys:
        data_map[key] = []
    for key in keys:
        for one_ins in batch:
            data_map[key].append(one_ins[key])
    for key in data_map:
        try:
            data_map[key] = pad_sequence(data_map[key], batch_first=True, padding_value=0)
        except:
            if data_map[key][0].size():
                raise ValueError(f"The {data_map[key]} can not be concat by pad_sequence.")
            data_map[key] = pad_sequence([i.unsqueeze(0) for i in data_map[key]], batch_first=True, padding_value=0).squeeze()


@datamodule_config_register("basic")
class BasicDatamodule(IBaseDataModule):
    def __init__(self, config: BasicDatamoduleConfig, data: Dict[str, pd.DataFrame]):
        super().__init__()

        self.key_type_pairs = config.key_type_pairs
        self.pin_memory = config.pin_memory
        self.shuffle: Dict[str, bool] = config.shuffle
        self.train_batch_size = config.train_batch_size
        self.valid_batch_size = config.valid_batch_size
        self.test_batch_size = config.test_batch_size
        self.predict_batch_size = config.predict_batch_size
        self.online_batch_size = config.online_batch_size
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.predict_data = None
        if 'train' in data:
            self.train_data = BasicDataset(self.key_type_pairs, data['train'])
        if "test" in data:
            self.test_data = BasicDataset(self.key_type_pairs, data['test'])
        if "valid" in data:
            self.valid_data = BasicDataset(self.key_type_pairs, data['valid'])
        if "predict" in data:
            self.predict_data = BasicDataset(self.key_type_pairs, data['predict'])
        self.collate_fn = base_collate

    # def train_dataloader(self):
        # return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def train_dataloader(self):
        if not self.train_data:
            return None
        return DataLoader(self.train_data, batch_size=self.train_batch_size, collate_fn=self.collate_fn, pin_memory=self.pin_memory)

    def predict_dataloader(self):
        """
        """
        if not self.predict_data:
            return None
        return DataLoader(self.predict_data, batch_size=self.predict_batch_size, collate_fn=self.collate_fn, pin_memory=self.pin_memory)

    def val_dataloader(self):
        if not self.valid_data:
            return None
        return DataLoader(self.valid_data, batch_size=self.valid_batch_size, collate_fn=self.collate_fn, pin_memory=self.pin_memory)

    def test_dataloader(self):
        if not self.test_data:
            return None
        return DataLoader(self.test_data, batch_size=self.test_batch_size, collate_fn=self.collate_fn, pin_memory=self.pin_memory)

    def online_dataloader(self):
        # return DataLoader(self.mnist_test, batch_size=self.batch_size)
        return self.collate_fn
