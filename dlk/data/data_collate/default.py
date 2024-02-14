# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict

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

from dlk.utils.register import register


@cregister("data_collate", "default")
class DefaultCollateConfig:
    """default collate function"""

    key_padding_pairs = DictField(
        value={}, help="the pair of key and padding value, the data is 1d"
    )
    key_padding_pairs_2d = DictField(
        value={}, help="the pair of key and padding value, the data is 2d"
    )
    key_no_padding = ListField(value=[], help="the key just concat no padding")
    key_padding_pairs_3d = DictField(
        value={}, help="the pair of key and padding value, the data is 3d"
    )
    gen_mask = DictField(value={}, help="the pair of key and generated mask key")


@register("data_collate", "default")
class DefaultCollate(object):
    """default collate function"""

    def __init__(self, config: DefaultCollateConfig):
        super(DefaultCollate, self).__init__()
        self.config = config

    def __call__(self, batch):
        keys = batch[0].keys()
        data_map: Dict[str, Any] = {}
        for key in keys:
            data_map[key] = []
        for key in keys:
            for one_ins in batch:
                data_map[key].append(one_ins[key])
        if self.config.gen_mask:
            for key, mask in self.config.gen_mask.items():
                if key not in data_map:
                    continue
                data_map[mask] = []
                for item in data_map[key]:
                    data_map[mask].append(
                        torch.tensor([1] * len(item), dtype=torch.int)
                    )
        for key in data_map:
            if key in self.config.key_no_padding:
                data = [ins for ins in data_map[key]]
                data_map[key] = torch.cat(data, dim=0)
            elif key in self.config.key_padding_pairs_3d:
                max_x, max_y, max_z = 0, 0, 0
                for ins in data_map[key]:
                    cur_x, cur_y, cur_z = ins.shape
                    max_x = max(max_x, cur_x)
                    max_y = max(max_y, cur_y)
                    max_z = max(max_z, cur_z)
                _data = torch.full(
                    (len(data_map[key]), max_x, max_y, max_z),
                    fill_value=self.config.key_padding_pairs_3d[key],
                    dtype=data_map[key][0].dtype,
                )
                for i, ins in enumerate(data_map[key]):
                    cur_x, cur_y, cur_z = ins.shape
                    _data[i][:cur_x, :cur_y, :cur_z] = ins
                data_map[key] = _data
            elif key in self.config.key_padding_pairs_2d:
                max_m, max_n = 0, 0
                for ins in data_map[key]:
                    cur_m, cur_n = ins.shape
                    max_m = max(max_m, cur_m)
                    max_n = max(max_n, cur_n)
                _data = torch.full(
                    (len(data_map[key]), max_m, max_n),
                    fill_value=self.config.key_padding_pairs_2d[key],
                    dtype=data_map[key][0].dtype,
                )
                for i, ins in enumerate(data_map[key]):
                    cur_m, cur_n = ins.shape
                    _data[i][:cur_m, :cur_n] = ins
                data_map[key] = _data
            elif key == "_index":
                _data = pad_sequence(
                    [i.unsqueeze(0) for i in data_map[key]],
                    batch_first=True,
                    padding_value=0,
                ).squeeze()
                if not _data.size():
                    _data.unsqueeze_(0)
                data_map[key] = _data
            else:
                try:
                    data_map[key] = pad_sequence(
                        data_map[key],
                        batch_first=True,
                        padding_value=self.config.key_padding_pairs.get(key, 0),
                    )
                except:
                    # if the data_map[key] is size 0, we can concat them
                    if data_map[key][0].size():
                        raise ValueError(
                            f"The {data_map[key]} can not be concat by pad_sequence."
                        )
                    _data = pad_sequence(
                        [i.unsqueeze(0) for i in data_map[key]],
                        batch_first=True,
                        padding_value=self.config.key_padding_pairs[key],
                    ).squeeze()
                    if not _data.size():
                        _data.unsqueeze_(0)
                    data_map[key] = _data
        return data_map
