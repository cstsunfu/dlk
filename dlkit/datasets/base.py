import pandas as pd
from torch.utils.data import DataLoader, random_split, Dataset
from typing import Dict, List, Union
from dlkit.utils.config import Config
from dlkit.datasets import dataset_register, dataset_config_register
import torch

@dataset_config_register('basic')
class BaseDatasetConfig(Config):
    """docstring for BasicDatasetConfig"""
    def __init__(self, **kwargs):
        super(BaseDatasetConfig, self).__init__(**kwargs)
        self.key_type_pair = kwargs.get('key_type_pair')
        # type_map is not configable
        self.type_map = {"float": torch.float, "int": torch.long, 'bool': torch.bool, "long": torch.long} 


class BaseDataset(Dataset):
    def __init__(self, config: BaseDatasetConfig, data:pd.DataFrame):
        self.key_type_pairs = config.key_type_pair
        self.type_map = config.type_map
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        one_ins = {}
        self.key_type_pairs: List
        for key, key_type in self.key_type_pairs:
            one_ins[key] = torch.tensor(self.data.iloc[idx][key], dtype=self.type_map[key_type])
        one_ins['_index'] = torch.tensor([idx],  dtype=torch.long)
        return one_ins

