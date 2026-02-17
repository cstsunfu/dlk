# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from intc import Base, BoolField, DictField, cregister
from torch.utils.data import Dataset

from dlk.utils.register import register

logger = logging.getLogger(__name__)


@cregister("dataset", "default")
class DefaultDatasetConfig(Base):
    """The default dataset configuration."""

    key_type_pairs = DictField(
        value={}, help="The pair of key and tensor type (e.g., 'input_ids': 'long')."
    )
    repeat_for_valid = BoolField(
        value=True,
        help="Whether to repeat the data for validation (useful for multi-card DDP validation consistency).",
    )


@register("dataset", "default")
class DefaultDataset(Dataset):
    """
    General Dataset supporting Pandas DataFrame, HuggingFace Datasets, and Dicts.
    """

    def __init__(
        self,
        config: DefaultDatasetConfig,
        data: Union[pd.DataFrame, HFDataset, Dict[str, List]],
        rt_config: Dict,
        key_type_pairs: Dict = None,
    ):
        """
        Args:
            config: Dataset configuration.
            data: Input data. Can be a Pandas DataFrame, HF Dataset (Arrow), or Dict (Online).
            rt_config: Runtime configuration (e.g., world_size).
            key_type_pairs: Optional override for key-type mapping.
        """
        self.config = config

        # Handle validation repetition for DDP (ensure batch sizes match across ranks if needed)
        self.repeat_valid = 1
        if self.config.repeat_for_valid and rt_config.get("world_size", 1) > 1:
            self.repeat_valid = rt_config.get("world_size", 1)

        self.data = data
        self.is_hf_dataset = isinstance(data, HFDataset)
        self.is_dict = isinstance(data, dict)
        self.is_pandas = isinstance(data, pd.DataFrame)

        # Tensor type mapping
        self.type_map = {
            "float": torch.float,
            "int": torch.int,
            "bool": torch.bool,
            "long": torch.long,
            "double": torch.double,
        }

        if key_type_pairs is not None:
            self.key_type_pairs = key_type_pairs
        else:
            # Dynamically determine available keys based on data type
            if self.is_hf_dataset:
                columns = data.column_names
            elif self.is_dict:
                columns = list(data.keys())
            elif self.is_pandas:
                columns = data.columns
            else:
                columns = []

            self.key_type_pairs = self.real_key_type_pairs(
                config.key_type_pairs, columns
            )

    @staticmethod
    def real_key_type_pairs(key_type_pairs: Dict, available_columns: List[str]):
        """Filter config keys to only include those present in the data."""
        return {k: v for k, v in key_type_pairs.items() if k in available_columns}

    def __len__(self):
        """Return the dataset size."""
        if self.is_hf_dataset or self.is_pandas:
            return len(self.data) * self.repeat_valid
        elif self.is_dict:
            # Assume all lists in dict have the same length
            first_key = next(iter(self.data))
            return len(self.data[first_key]) * self.repeat_valid
        return 0

    def __getitem__(self, idx: int):
        """Return one instance by index converted to tensors."""
        # Handle repetition logic
        real_idx = idx // self.repeat_valid

        # Fetch data based on storage backend
        if self.is_hf_dataset:
            # HF Dataset access is optimized
            item = self.data[real_idx]
        elif self.is_dict:
            # Dict access
            item = {
                k: v[real_idx] for k, v in self.data.items() if k in self.key_type_pairs
            }
        elif self.is_pandas:
            # Pandas access (legacy, slower)
            item = self.data.iloc[real_idx]
        else:
            raise TypeError("Unsupported data type in Dataset.")

        return self.prepare_tensor(real_idx, item)

    def prepare_tensor(self, idx: int, item: Dict[str, Any]):
        """Convert raw data items to PyTorch tensors based on config."""
        one_ins = {}
        for key, key_type in self.key_type_pairs.items():
            value = item[key]

            if key_type in ("sparse", "object"):
                one_ins[key] = value
                continue

            if key_type not in self.type_map:
                raise ValueError(
                    f"Unknown key_type: '{key_type}' defined for key: '{key}'"
                )

            # Convert to numpy first for efficiency/safety, then to tensor
            # HF Datasets usually return python objects or numpy arrays
            try:
                if isinstance(value, (list, tuple)):
                    tensor_val = torch.tensor(value, dtype=self.type_map.get(key_type))
                elif isinstance(value, np.ndarray):
                    tensor_val = torch.from_numpy(value)
                    if key_type in self.type_map:
                        tensor_val = tensor_val.to(self.type_map[key_type])
                else:
                    # Scalars
                    tensor_val = torch.tensor(value, dtype=self.type_map.get(key_type))

                one_ins[key] = tensor_val
            except Exception as e:
                logger.error(
                    f"Error converting key '{key}' with value '{value}' to type '{key_type}': {e}"
                )
                raise e

        # Add index for tracking
        one_ins["_index"] = torch.tensor(idx, dtype=torch.long)
        return one_ins
