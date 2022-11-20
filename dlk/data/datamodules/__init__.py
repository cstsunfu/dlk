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

"""datamodules"""

import importlib
import os
from typing import Dict, Any
from dlk.utils.register import Register
from pytorch_lightning import LightningDataModule
import abc
from torch.nn.utils.rnn import pad_sequence
import torch

datamodule_config_register = Register("Datamodule config register")
datamodule_register = Register("Datamodule register")


collate_register = Register('Collate function register.')



@collate_register('default')
class DefaultCollate(object):
    """docstring for DefaultCollate"""
    def __init__(self, **config):
        super(DefaultCollate, self).__init__()
        self.key_padding_pairs = config.get("key_padding_pairs", {})
        self.key_padding_pairs_2d = config.get("key_padding_pairs_2d", {})
        self.key_padding_pairs_3d = config.get("key_padding_pairs_3d", {})
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
            for key, mask in self.gen_mask.items():
                if key not in data_map:
                    continue
                data_map[mask] = []
                for item in data_map[key]:
                    data_map[mask].append(torch.tensor([1] * len(item), dtype=torch.int))
        for key in data_map:
            if key in self.key_padding_pairs_3d:
                max_x, max_y, max_z = 0, 0, 0
                for ins in data_map[key]:
                    cur_x, cur_y, cur_z = ins.shape
                    max_x = max(max_x, cur_x)
                    max_y = max(max_y, cur_y)
                    max_z = max(max_z, cur_z)
                _data = torch.full((len(data_map[key]), max_x, max_y, max_z), fill_value=self.key_padding_pairs_3d[key], dtype=data_map[key][0].dtype)
                for i, ins in enumerate(data_map[key]):
                    cur_x, cur_y, cur_z = ins.shape
                    _data[i][:cur_x,:cur_y, :cur_z] = ins
                data_map[key] = _data
            elif key in self.key_padding_pairs_2d:
                max_m, max_n = 0, 0
                for ins in data_map[key]:
                    cur_m, cur_n = ins.shape
                    max_m = max(max_m, cur_m)
                    max_n = max(max_n, cur_n)
                _data = torch.full((len(data_map[key]), max_m, max_n), fill_value=self.key_padding_pairs_2d[key], dtype=data_map[key][0].dtype)
                for i, ins in enumerate(data_map[key]):
                    cur_m, cur_n = ins.shape
                    _data[i][:cur_m,:cur_n] = ins
                data_map[key] = _data
            elif key == '_index':
                _data = pad_sequence([i.unsqueeze(0) for i in data_map[key]], batch_first=True, padding_value=0).squeeze()
                if not _data.size():
                    _data.unsqueeze_(0)
                data_map[key] = _data
            else:
                try:
                    data_map[key] = pad_sequence(data_map[key], batch_first=True, padding_value=self.key_padding_pairs.get(key, 0))
                except:
                    # if the data_map[key] is size 0, we can concat them
                    if data_map[key][0].size():
                        raise ValueError(f"The {data_map[key]} can not be concat by pad_sequence.")
                    _data = pad_sequence([i.unsqueeze(0) for i in data_map[key]], batch_first=True, padding_value=self.key_padding_pairs[key]).squeeze()
                    if not _data.size():
                        _data.unsqueeze_(0)
                    data_map[key] = _data
        return data_map


class IBaseDataModule(LightningDataModule):
    """docstring for IBaseDataModule"""
    def __init__(self):
        super(IBaseDataModule, self).__init__()

    def train_dataloader(self):
        """

        Raises:
            NotImplementedError

        """
        raise NotImplementedError(f"You must implementation the train_dataloader for your own datamodule.")

    def predict_dataloader(self):
        """

        Raises:
            NotImplementedError

        """
        raise NotImplementedError(f"You must implementation the predict_dataloader for your own datamodule.")

    def val_dataloader(self):
        """

        Raises:
            NotImplementedError

        """
        raise NotImplementedError(f"You must implementation the val_dataloader for your own datamodule.")

    def test_dataloader(self):
        """

        Raises:
            NotImplementedError

        """
        raise NotImplementedError(f"You must implementation the test_dataloader for your own datamodule.")

    @abc.abstractmethod
    def online_dataloader(self):
        """

        Raises:
            NotImplementedError

        """
        raise NotImplementedError(f"You must implementation the online_dataloader for your own datamodule.")


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
import_datamodules(datamodules_dir, "dlk.data.datamodules")
