import pandas as pd
from torch.utils.data import DataLoader, random_split, Dataset
from typing import Dict, List, Union
import torch
'''
dataset
{
    "_name": "base",
    "config": {
        "key_type_pair": [('x', 'float'), ('y', 'int')],
        'pin_memory': false //if use cuda, this should set true
    }
}

'''


'''
datamodule

{
    "_name": "base",
    "dataloader@train": {
        "_base": "base",
        "collate": {
            "_name": "base",
            "config": {
                "padding": 0,
            }
        }

        "config": {
            'pin_memory': false //if use cuda, this should set true
        }
    },
    "dataset@train": {
        "_name": "base",
        "config": {
            "key_type_pair": [('x', 'float'), ('y', 'int')],
        }
    },
    "config": {
        "datasets": {
            "train": {
                "data": "train"
            },
        }
    }
}
'''
class BaseDataset(Dataset):
    def __init__(self, config: Dict[str, str], data:pd.DataFrame):
        self.key_type_pairs = config.get('key_type_pair')
        self.type_map = {"float": torch.float, "int": torch.long, 'bool': torch.bool, "long": torch.long}
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

