# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List

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
    key_pass_through = ListField(
        value=[],
        suggestions=[["sentence"]],
        help="the key just pass through, no padding, no concat",
    )
    gen_mask = DictField(value={}, help="the pair of key and generated mask key")


@register("data_collate", "default")
class DefaultCollate(object):
    """Default collate function handling dynamic padding, stacking, and masking."""

    def __init__(self, config: DefaultCollateConfig):
        super(DefaultCollate, self).__init__()
        self.config = config

    def __call__(
        self, batch: List[Dict[str, Any]], stage: str = "train"
    ) -> Dict[str, torch.Tensor]:
        if not batch:
            return {}

        keys = batch[0].keys()
        data_map: Dict[str, Any] = {key: [] for key in keys}

        # Transpose list of dicts to dict of lists
        for one_ins in batch:
            for key in keys:
                data_map[key].append(one_ins[key])

        # Automatic mask generation
        if self.config.gen_mask:
            for source_key, mask_key in self.config.gen_mask.items():
                if source_key not in data_map:
                    continue
                data_map[mask_key] = [
                    torch.ones(len(item), dtype=torch.int)
                    for item in data_map[source_key]
                ]

        # Padding and Stacking Logic
        for key in data_map:
            if key in self.config.key_pass_through:
                continue
            if key in self.config.key_no_padding:
                # Direct concatenation without padding
                data_map[key] = torch.cat(data_map[key], dim=0)

            elif key in self.config.key_padding_pairs_3d:
                # 3D padding (e.g., video or volume data)
                max_x, max_y, max_z = 0, 0, 0
                for ins in data_map[key]:
                    cur_x, cur_y, cur_z = ins.shape
                    max_x, max_y, max_z = (
                        max(max_x, cur_x),
                        max(max_y, cur_y),
                        max(max_z, cur_z),
                    )

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
                # 2D padding (e.g., span matrices)
                max_m, max_n = 0, 0
                for ins in data_map[key]:
                    cur_m, cur_n = ins.shape
                    max_m, max_n = max(max_m, cur_m), max(max_n, cur_n)

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
                # Special internal index tracking
                data_map[key] = torch.stack(data_map[key], dim=0)

            else:
                # 1D/Standard Padding for sequences or Stacking for scalars
                if len(data_map[key]) > 0:
                    if (
                        isinstance(data_map[key][0], torch.Tensor)
                        and data_map[key][0].dim() == 0
                    ):
                        # Handle scalars (0D tensors) efficiently via stack
                        data_map[key] = torch.stack(data_map[key], dim=0)
                    elif not isinstance(data_map[key][0], torch.Tensor):
                        pass
                    else:
                        try:
                            data_map[key] = pad_sequence(
                                data_map[key],
                                batch_first=True,
                                padding_value=self.config.key_padding_pairs.get(key, 0),
                            )
                        except RuntimeError as e:
                            shapes = [tuple(t.shape) for t in data_map[key]]
                            raise RuntimeError(
                                f"Shape mismatch error while padding sequence for key '{key}'.\n"
                                f"Detected shapes: {shapes}.\nOriginal Exception: {e}"
                            ) from e
                else:
                    try:
                        data_map[key] = pad_sequence(
                            data_map[key],
                            batch_first=True,
                            padding_value=self.config.key_padding_pairs.get(key, 0),
                        )
                    except RuntimeError as e:
                        # Raised when tensors have inconsistent shapes in dimensions other than the sequence dim
                        shapes = [tuple(t.shape) for t in data_map[key]]
                        raise RuntimeError(
                            f"Shape mismatch error while padding sequence for key '{key}'.\n"
                            f"Ensure all non-sequence dimensions match.\n"
                            f"Detected shapes in batch: {shapes}.\n"
                            f"Original Exception: {e}"
                        ) from e
                    except TypeError as e:
                        # Raised when data_map contains objects that are not PyTorch Tensors
                        types = [
                            type(t).__name__ for t in data_map[key][:3]
                        ]  # Check first few items
                        raise TypeError(
                            f"Type mismatch error while padding key '{key}'.\n"
                            f"Expected torch.Tensor, but found types like: {types}.\n"
                            f"Original Exception: {e}"
                        ) from e

        return data_map
