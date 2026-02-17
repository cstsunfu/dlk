# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional, Type

import lightning.pytorch as pl
import torch
from datasets import Dataset, load_from_disk
from intc import AnyField, Base, BoolField, IntField, SubModule, cregister
from torch.utils.data import DataLoader

from dlk.data.datamodule import IBaseDataModule
from dlk.data.dataset.default import DefaultDataset, DefaultDatasetConfig
from dlk.utils.register import register, register_module_name

logger = logging.getLogger(__name__)


@cregister("datamodule", "default")
class BasicDatamoduleConfig(Base):
    """The default datamodule configuration."""

    pin_memory = AnyField(
        value=None,
        options=[None, True, False],
        help="Whether to use pin_memory. If None, checks for GPU availability.",
    )
    num_workers = IntField(
        value=1,
        minimum=-1,
        help="The number of workers for dataloader. -1 uses os.cpu_count().",
    )
    shuffle = BoolField(value=True, help="Whether to shuffle the training data.")
    train_batch_size = IntField(
        value=32, minimum=1, help="The batch size of train dataloader."
    )
    predict_batch_size = IntField(
        value=32, minimum=1, help="The batch size of predict dataloader."
    )
    online_batch_size = IntField(
        value=1, minimum=1, help="The batch size of online dataloader."
    )

    drop_last = BoolField(
        value=False,
        help="for contrastive learning or other task need full batch size training, set drop_last to true",
    )

    submodule = SubModule({}, suggestions=["dataset", "data_collate"])


@register("datamodule", "default")
class BasicDatamodule(IBaseDataModule):
    """
    Basic and General DataModule supporting Distributed Training (DDP) and Online Inference.
    """

    def __init__(
        self,
        config: BasicDatamoduleConfig,
        data: Optional[Dict[str, Any]],
        rt_config: Dict,
    ):
        """
        Args:
            config: DataModule config.
            data: Dictionary of data paths or loaded data objects.
                  In Phase 4, this is primarily used to pass metadata or overrides.
                  Actual heavy data loading happens in `setup()`.
            rt_config: Runtime config (e.g. processed_data_dir, world_size).
        """
        super().__init__()

        self.config = config
        self.rt_config = rt_config
        self.data_container = data if data else {}

        # Placeholders for datasets
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.predict_data = None

        # Dataset Factory
        self.dataset_config: DefaultDatasetConfig = config.submodule.dataset
        self.dataset_creator: Type[DefaultDataset] = register.get(
            "dataset", register_module_name(self.dataset_config._module_name)
        )

        # Collate Function
        data_collate_config = config.submodule.data_collate
        self.collate_fn = register.get(
            "data_collate", register_module_name(data_collate_config._module_name)
        )(data_collate_config)

        # Worker configuration
        if self.config.num_workers == -1:
            self.config.num_workers = os.cpu_count() or 1

    def prepare_data(self):
        """
        Use this to download and process data.
        In distributed training, this is called only on rank 0.

        Note: In DLK architecture, heavy processing usually happens in `processor.fit()`
        before `train.py` runs `DataModule`. So this method is mostly a placeholder
        or for last-mile checks.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set state (self.train_data, self.valid_data, etc.).
        In distributed training, this is called on every GPU/Node.

        We use `load_from_disk` which uses memory mapping (Zero-Copy), so multiple
        processes sharing the same data file do not consume extra RAM.
        """
        processed_dir = self.rt_config.get("processed_data_dir", "data/processed_data")

        # Determine paths (priority: provided in __init__ -> constructed from processed_data_dir)
        train_path = self.data_container.get(
            "train", os.path.join(processed_dir, "train")
        )
        valid_path = self.data_container.get(
            "valid", os.path.join(processed_dir, "valid")
        )
        test_path = self.data_container.get("test", os.path.join(processed_dir, "test"))
        predict_path = self.data_container.get(
            "predict", os.path.join(processed_dir, "predict")
        )

        def _get_dataset(data_obj, default_path):
            from datasets import Dataset

            if isinstance(data_obj, (Dataset, list, dict)):
                return data_obj

            path = data_obj if isinstance(data_obj, str) else default_path
            if os.path.exists(path) and (
                os.path.exists(os.path.join(path, "dataset_info.json"))
                or os.path.exists(os.path.join(path, "state.json"))
            ):
                try:
                    return load_from_disk(path)
                except Exception as e:
                    logger.error(f"Failed to load dataset from {path}: {e}")
            return None

        if stage == "fit" or stage is None:
            if not self.train_data:
                hf_train = _get_dataset(
                    train_path, os.path.join(processed_dir, "train")
                )
                if hf_train is not None:
                    logger.info(f"Loading Train data, size: {len(hf_train)}")
                    self.train_data = self.dataset_creator(
                        self.dataset_config, hf_train, self.rt_config
                    )

            if not self.valid_data:
                hf_valid = _get_dataset(
                    valid_path, os.path.join(processed_dir, "valid")
                )
                if hf_valid is not None:
                    logger.info(f"Loading Valid data, size: {len(hf_valid)}")
                    self.valid_data = self.dataset_creator(
                        self.dataset_config, hf_valid, self.rt_config
                    )

        if stage == "test" or stage is None:
            if not self.test_data:
                hf_test = _get_dataset(test_path, os.path.join(processed_dir, "test"))
                if hf_test is not None:
                    logger.info(f"Loading Test data, size: {len(hf_test)}")
                    self.test_data = self.dataset_creator(
                        self.dataset_config, hf_test, self.rt_config
                    )

        if stage == "predict":
            if not self.predict_data:
                hf_predict = _get_dataset(
                    predict_path, os.path.join(processed_dir, "predict")
                )
                if hf_predict is not None:
                    logger.info(f"Loading Predict data, size: {len(hf_predict)}")
                    self.predict_data = self.dataset_creator(
                        self.dataset_config, hf_predict, self.rt_config
                    )

    def train_dataloader(self):
        """Get the train set dataloader."""
        if not self.train_data:
            return None
        return DataLoader(
            self.train_data,
            batch_size=self.config.train_batch_size,
            collate_fn=partial(self.collate_fn, stage="train"),
            pin_memory=(
                self.config.pin_memory
                if self.config.pin_memory is not None
                else torch.cuda.is_available()
            ),
            drop_last=self.config.drop_last,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.num_workers > 0,
        )

    def val_dataloader(self):
        """Get the validation set dataloader."""
        if not self.valid_data:
            return None
        return DataLoader(
            self.valid_data,
            batch_size=self.config.predict_batch_size,
            collate_fn=partial(self.collate_fn, stage="valid"),
            drop_last=self.config.drop_last,
            pin_memory=(
                self.config.pin_memory
                if self.config.pin_memory is not None
                else torch.cuda.is_available()
            ),
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=self.config.num_workers > 0,
        )

    def test_dataloader(self):
        """Get the test set dataloader."""
        if not self.test_data:
            return None
        return DataLoader(
            self.test_data,
            batch_size=self.config.predict_batch_size,
            collate_fn=partial(self.collate_fn, stage="test"),
            drop_last=self.config.drop_last,
            pin_memory=(
                self.config.pin_memory
                if self.config.pin_memory is not None
                else torch.cuda.is_available()
            ),
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def predict_dataloader(self):
        """Get the predict set dataloader."""
        if not self.predict_data:
            return None
        return DataLoader(
            self.predict_data,
            batch_size=self.config.predict_batch_size,
            collate_fn=partial(self.collate_fn, stage="predict"),
            drop_last=self.config.drop_last,
            pin_memory=(
                self.config.pin_memory
                if self.config.pin_memory is not None
                else torch.cuda.is_available()
            ),
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def online_dataloader(self, data: Dict[str, List]):
        """
        Get the online dataloader for a specific batch of data (Dict format).
        Used for legacy compatibility or when DataLoader abstraction is strictly needed.
        """
        columns = list(data.keys())
        key_type_pairs = self.dataset_creator.real_key_type_pairs(
            self.dataset_config.key_type_pairs, columns
        )
        dataset = self.dataset_creator(
            self.dataset_config, data, self.rt_config, key_type_pairs
        )
        return DataLoader(
            dataset,
            batch_size=self.config.online_batch_size,
            collate_fn=partial(self.collate_fn, stage="online"),
            pin_memory=False,  # Online usually immediate
            shuffle=False,
            num_workers=0,  # Avoid overhead for small batches
        )

    def online_process_batch(
        self, data: Dict[str, List[Any]]
    ) -> Dict[str, torch.Tensor]:
        """Directly process a batch dictionary into model inputs bypassing DataLoader.

        This method is optimized for low-latency inference servers.

        Args:
            data: A dictionary of lists, e.g. {"input_ids": [[...], [...]], "mask": ...}

        Returns:
            A batch dictionary containing Tensors ready for the model.
        """
        columns = list(data.keys())
        key_type_pairs = self.dataset_creator.real_key_type_pairs(
            self.dataset_config.key_type_pairs, columns
        )

        batch_size = len(next(iter(data.values())))
        batch_items = [{} for _ in range(batch_size)]

        type_map = {
            "float": torch.float,
            "int": torch.int,
            "bool": torch.bool,
            "long": torch.long,
            "double": torch.double,
        }

        for key, key_type in key_type_pairs.items():
            values = data[key]
            for i in range(batch_size):
                val = values[i]
                if key_type in ("sparse", "object"):
                    batch_items[i][key] = val
                    continue

                target_dtype = type_map.get(key_type)
                try:
                    if isinstance(val, (list, tuple)):
                        batch_items[i][key] = torch.tensor(val, dtype=target_dtype)
                    elif isinstance(val, np.ndarray):
                        tensor_val = torch.from_numpy(val)
                        if target_dtype:
                            tensor_val = tensor_val.to(target_dtype)
                        batch_items[i][key] = tensor_val
                    else:
                        batch_items[i][key] = torch.tensor(val, dtype=target_dtype)
                except Exception as e:
                    logger.error(
                        f"Error converting key '{key}' to type '{key_type}': {e}"
                    )
                    raise ValueError(f"Conversion failed for {key}") from e

        for i in range(batch_size):
            batch_items[i]["_index"] = torch.tensor(i, dtype=torch.long)

        batch = self.collate_fn(batch_items, stage="online")
        return batch
