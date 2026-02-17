# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import logging
import os
import pickle as pkl
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from intc import (
    MISSING,
    Base,
    BoolField,
    IntField,
    ListField,
    NestField,
    StrField,
    SubModule,
    cregister,
)

from dlk.utils.io import open
from dlk.utils.register import register

logger = logging.getLogger(__name__)


@cregister("processor", "default")
class DefaultProcessorConfig(Base):
    """The default processor configuration."""

    data_root = StrField(
        value=None,
        additions=[None],
        help="The root path of input data. If set, all data paths will be relative to this root.",
    )
    train_data_type = StrField(
        value="none",
        options=[
            "dict",
            "dataframe",
            "dataset",  # NEW: Support passing datasets.Dataset object directly
            "parquet",
            "parquet_list",
            "json",
            "arrow",
            "disk",
            "pickle",
            "none",
        ],
        help="The type of train data. `dataset` means passing a HF Dataset object in python script.",
    )
    valid_data_type = StrField(
        value="none",
        options=[
            "dict",
            "dataframe",
            "dataset",
            "parquet",
            "json",
            "arrow",
            "disk",
            "pickle",
            "none",
        ],
        help="The type of valid data.",
    )
    test_data_type = StrField(
        value="none",
        options=[
            "dict",
            "dataframe",
            "dataset",
            "parquet",
            "json",
            "arrow",
            "disk",
            "pickle",
            "none",
        ],
        help="The type of test data.",
    )
    predict_data_type = StrField(
        value="none",
        options=[
            "dict",
            "dataframe",
            "dataset",
            "parquet",
            "parquet_list",
            "json",
            "arrow",
            "disk",
            "pickle",
            "none",
        ],
        help="The type of predict data.",
    )
    online_data_type = StrField(
        value="none",
        options=["dict", "dataframe", "none"],
        help="The type of online data. Usually 'dict' for server requests.",
    )
    feed_order = ListField(value=[], help="The order of subprocessors execution.")
    meta_collection_on_train = BoolField(
        value=True, help="Whether to collect meta info on train data."
    )
    load_meta_on_start = BoolField(
        value=False, help="Whether to load meta info when starting the processor."
    )
    processed_data_dir = StrField(
        value="data/processed_data", help="The directory to save processed data."
    )
    meta_dir = StrField(value="data/meta_data", help="The directory to save meta info.")
    do_save = BoolField(value=True, help="Whether to save the processed data to disk.")
    num_proc = IntField(value=4, minimum=1, help="Number of processes to use.")
    verbose = BoolField(
        value=True, help="Whether to print verbose logs during processing."
    )
    submodule = SubModule(value={}, help="Subprocessors configuration.")


def load_data_source(
    origin: Any, data_type: str, config: DefaultProcessorConfig
) -> Iterator[Dataset]:
    """Load data from various sources and yield HuggingFace Datasets."""
    if data_type == "none":
        yield None
        return

    # Handle data_root logic for paths
    if config.data_root and isinstance(origin, (str, list)):
        if isinstance(origin, str):
            origin = os.path.join(config.data_root, origin)
        elif isinstance(origin, list):
            origin = [os.path.join(config.data_root, p) for p in origin]

    if data_type == "dataset":
        yield origin
    elif data_type == "dataframe":
        yield Dataset.from_pandas(origin)
    elif data_type == "dict":
        yield Dataset.from_dict(origin)
    elif data_type == "pickle":
        df = pkl.load(open(origin, "rb"))
        yield Dataset.from_pandas(df)
    elif data_type == "json":
        yield load_dataset("json", data_files=origin, split="train")
    elif data_type == "parquet":
        yield load_dataset("parquet", data_files=origin, split="train")
    elif data_type == "parquet_list":
        yield load_dataset("parquet", data_files=origin, split="train")
    elif data_type in ["arrow", "disk"]:
        yield load_from_disk(origin)
    else:
        raise NotImplementedError(f"Data type {data_type} is not supported.")


@register("processor", "default")
class DefaultProcessor(object):

    stage_data_set_map = {
        "collect": "collect_data_set",
        "train": "train_data_set",
        "predict": "predict_data_set",
        "online": "online_data_set",
    }

    def __init__(self, stage: str, config: DefaultProcessorConfig):
        super(DefaultProcessor, self).__init__()
        config_dict = config._to_dict()
        self.stage = stage
        self.config: DefaultProcessorConfig = config

        self.subprocessors = {}

        for name in self.config.feed_order:
            subprocessor_config_dict = config_dict.get(f"@subprocessor@{name}")
            if not subprocessor_config_dict:
                logger.warning(f"Config for subprocessor '{name}' not found.")
                continue
            subprocessor_name = subprocessor_config_dict["_name"].split("-")[0]
            subprocessor_config = cregister.get("subprocessor", subprocessor_name)(
                subprocessor_config_dict
            )
            subprocessor = register.get("subprocessor", subprocessor_name)(
                stage=self.stage,
                config=subprocessor_config,
                meta_dir=self.config.meta_dir,
            )

            if self.config.load_meta_on_start:
                subprocessor.load_meta()
            used_data_set_names = set(
                subprocessor_config[self.stage_data_set_map[self.stage]]
            )
            if self.config.meta_collection_on_train and self.stage == "train":
                update_data_set_names = set(
                    subprocessor_config[self.stage_data_set_map["collect"]]
                )
                used_data_set_names = used_data_set_names.union(update_data_set_names)
            for data_set_name in used_data_set_names:
                if data_set_name not in self.subprocessors:
                    self.subprocessors[data_set_name] = []
                self.subprocessors[data_set_name].append(
                    (subprocessor, subprocessor_name)
                )

    def get_cache_path(self, type_name: str) -> str:
        return os.path.join(self.config.processed_data_dir, type_name)

    def get_config_hash(self):
        """Generates a stable MD5 hash based on the current configuration."""
        dict_repr = json.dumps(self.config._to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.md5(dict_repr).hexdigest()

    def is_cached(self, type_name: str) -> bool:
        """Validates if the cached dataset is exactly aligned with current configuration."""
        path = self.get_cache_path(type_name)
        hash_path = os.path.join(path, "config_hash.txt")
        if not os.path.exists(path) or not os.path.exists(hash_path):
            return False
        with open(hash_path, "r") as f:
            saved_hash = f.read().strip()
        return saved_hash == self.get_config_hash()

    def process(self, data: Dict) -> Dict:
        result = {}
        if not self.config.do_save:
            result = {key: [] for key in data}

        for type_name in ["train", "valid", "test", "predict"]:
            cache_path = self.get_cache_path(type_name)

            # Use strict MD5 Hash Check!
            if self.config.do_save and self.is_cached(type_name):
                logger.info(f"Load strict cached {type_name} from {cache_path}")
                try:
                    result[type_name] = load_from_disk(cache_path)
                    continue
                except:
                    pass

            processed_datasets = []
            type_datasets = data.get(type_name, [])
            if isinstance(type_datasets, Dataset):
                type_datasets = [type_datasets]
            else:
                assert isinstance(
                    type_datasets, list
                ), f"Expected data[{type_name}] to be a Dataset or list of Datasets, got {type(type_datasets)}"
            for i, dataset in enumerate(type_datasets):
                deliver_meta = (
                    self.config.meta_collection_on_train
                    and type_name == "train"
                    and i == 0
                )

                for sub_processor, sub_processor_name in self.subprocessors.get(
                    type_name, []
                ):
                    if self.config.verbose:
                        logger.info(f"Processing {type_name}: {sub_processor_name}")

                    dataset = sub_processor.process(
                        data=dataset,
                        deliver_meta=deliver_meta,
                        num_proc=self.config.num_proc,
                    )
                processed_datasets.append(dataset)

            if processed_datasets:
                if len(processed_datasets) > 1:
                    final_dataset = concatenate_datasets(processed_datasets)
                else:
                    final_dataset = processed_datasets[0]

                if self.config.do_save:
                    final_dataset.save_to_disk(cache_path)

                    # Store MD5 Hash explicitly
                    with open(os.path.join(cache_path, "config_hash.txt"), "w") as f:
                        f.write(self.get_config_hash())

                    result[type_name] = load_from_disk(cache_path)
                else:
                    result[type_name] = final_dataset

        return result

    def online_process(self, data: Dict) -> Dict:

        for sub_processor, sub_processor_name in self.subprocessors.get("online", []):
            data = sub_processor.process(data, deliver_meta=False)
        return data
