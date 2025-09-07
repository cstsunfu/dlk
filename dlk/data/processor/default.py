# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pickle as pkl
from typing import Callable, Dict, Iterator, Type

import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset
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

from dlk.utils.register import register

logger = logging.getLogger(__name__)


@cregister("processor", "default")
class DefaultProcessorConfig(Base):
    """the default processor"""

    data_root = StrField(
        value=None,
        additions=[None],
        help="the root path of input(should be processed) data, if set the data_root to not null, all the path of data will be relative to the data_root",
    )
    feed_order = ListField(
        value=[],
        suggestions=[["tokenizer", "token_gather", "label_to_id", "token_embedding"]],
        help="the order of data feed",
    )
    meta_collection_on_train = BoolField(
        value=True,
        help="whether to collect meta info on train data, if there are more than one train data, we will only collect meta info on the first part.",
    )
    load_meta_on_start = BoolField(
        value=False,
        help="whether to load meta info when start the processor, when `load_meta_on_start` set to `True`, the `meta_collection_on_train` must be `False`.",
    )
    processed_data_dir = StrField(
        value="data/processed_data",
        help="the save dir of the processor, not effective by `data_root`.",
    )
    meta_dir = StrField(
        value="data/meta_data",
        help="the save dir of the meta info, not effective by `data_root`.",
    )
    batch_size = IntField(
        value=1000,
        help="the batch size to process the data, only effective when the data is a datasets.Dataset",
    )
    collect_batch_size = FloatField(
        value=1.0,
        minimum=0.0,
        maximum=1.0,
        help="the batch size to collect meta info, only effective when the data is a datasets.Dataset",
    )
    num_proc = IntField(
        value=-1,
        help="the number of process to process the data, only effective when the data is a datasets.Dataset, -1 means cpu_count",
    )
    do_save = BoolField(
        value=True,
        help="""
        whether save the processed data, 
        if `false` will return the processed dict
        """,
    )

    submodule = SubModule(
        value={},
        help="subprocessors for processor",
    )


@register("processor", "default")
class DefaultProcessor(object):
    """docstring for IProcessor"""

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

        if self.config.num_proc == -1:
            self.config.num_proc = os.cpu_count()

        assert (not self.config.meta_collection_on_train) or (
            not self.config.load_meta_on_start
        )

        self.subprocessors = {}  # for dataset type

        for name in self.config.feed_order:
            subprocessor_config_dict = config_dict[f"@subprocessor@{name}"]
            if not subprocessor_config_dict[self.stage_data_set_map[stage]]:
                logger.info(f"Skip '{name}' ....")
                continue
            logger.info(f"Init '{name}' ....")
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

            for data_set in subprocessor_config_dict[self.stage_data_set_map[stage]]:
                if data_set not in self.subprocessors:
                    self.subprocessors[data_set] = []
                self.subprocessors[data_set].append(subprocessor)

    def save(self, dataset, dataset_name: str):
        """save data to self.config.processed_data_dir

        Args:
            data: should saved data

        Returns:
            None
        """
        dataset.save_to_disk(os.path.join(self.config.processed_data_dir, dataset_name))

    @classmethod
    def do_collect(cls, config, datasets):
        """directly collect meta info on any stage
        Returns:
            None
        """
        processor = cls(stage="collect", config=config)
        processor.process(datasets)

    def process(self, datasets: Dict) -> Dict:
        """Process entry

        Args:
            datasets:
            >>> {
            >>>     "train": {training data....},
            >>>     "test": ..
            >>> }

        Returns:
            processed data
        """
        if self.config.meta_collection_on_train and self.stage == "train":
            self.do_collect(self.config, datasets)
        result = {}
        for dataset_name in datasets:
            processed_data: Dataset = datasets[dataset_name]
            for subprocessor in self.subprocessors.get(dataset_name, []):
                processed_data = processed_data.map(
                    lambda x: subprocessor.process(data=x),
                    batched=True,
                    batch_size=self.config.batch_size
                    if self.stage != "collect"
                    else int(self.config.collect_batch_size * len(processed_data)),
                    num_proc=self.config.num_proc,
                )
            if not self.config.do_save:
                result[dataset_name].append(processed_data)
            else:
                self.save(processed_data, dataset_name)
        return result

    def online_process(self, data: Dict):
        """online server process the data without save
        Args:
            data:
                the data to be processed

        Returns:
            processed data
        """
        for subprocessor in self.subprocessors.get("online", []):
            data = subprocessor.process(data=data)
        return data
