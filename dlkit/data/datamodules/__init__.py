"""datamodules"""

import importlib
import os
from typing import Dict, Any
from dlkit.utils.register import Register
from pytorch_lightning import LightningDataModule
import abc
from torch.nn.utils.rnn import pad_sequence

datamodule_config_register = Register("Datamodule config register")
datamodule_register = Register("Datamodule register")


collate_register = Register('Collate function register.')



@collate_register('default')
class DefaultCollate(object):
    """docstring for DefaultCollate"""
    def __init__(self, **config):
        super(DefaultCollate, self).__init__()
        self.key_padding_pairs = config.get("key_padding_pairs", {})
        self.gen_mask = config.get("gen_mask", {})

    def __call__(self, batch):
        keys = batch[0].keys()
        data_map: Dict[str, Any] = {}
        for key in keys:
            data_map[key] = []
        for key in keys:
            for one_ins in batch:
                data_map[key].append(one_ins[key])
        if self.gen_mask:
            assert len(self.gen_mask) == 1
            key = list(self.gen_mask)[0]
            data_map[self.gen_mask[key]] = []
            for item in data_map[key]:
                data_map[self.gen_mask[key]].append([1] * len(item))
        for key in data_map:
            try:
                data_map[key] = pad_sequence(data_map[key], batch_first=True, padding_value=self.key_padding_pairs.get(key, 0))
            except:
                # if the data_map[key] is size 0, we can concat them
                if data_map[key][0].size():
                    raise ValueError(f"The {data_map[key]} can not be concat by pad_sequence.")
                _data = pad_sequence([i.unsqueeze(0) for i in data_map[key]], batch_first=True, padding_value=self.key_padding_pairs.get(key, 0)).squeeze()
                if not _data.size():
                    _data.unsqueeze_(0)
                data_map[key] = _data
        return data_map


class IDataModule(metaclass=abc.ABCMeta):
    """docstring for ModuleStepMixin"""


    @abc.abstractmethod
    def train_dataloader(self):
        pass

    @abc.abstractmethod
    def predict_dataloader(self):
        pass

    @abc.abstractmethod
    def val_dataloader(self):
        pass

    @abc.abstractmethod
    def test_dataloader(self):
        pass

    @abc.abstractmethod
    def online_dataloader(self):
        pass


class IBaseDataModule(LightningDataModule, IDataModule):
    """docstring for IBaseDataModule"""
    def __init__(self):
        super(IBaseDataModule, self).__init__()
        

def import_datamodules(datamodules_dir, namespace):
    for file in os.listdir(datamodules_dir):
        path = os.path.join(datamodules_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            datamodule_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + datamodule_name)


# automatically import any Python files in the models directory
datamodules_dir = os.path.dirname(__file__)
import_datamodules(datamodules_dir, "dlkit.data.datamodules")
