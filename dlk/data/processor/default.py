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
    train_data_type = StrField(
        value="none",
        options=[
            "dict",
            "dataframe",
            "parquet",
            "parquet_list",
            "json",
            "none",
            "pickle",
        ],
        help="the type of train data, `none` means no train data, `parquet_list` means the train data is a list of parquet file path, we will separate load->process->save the data into several parts, `pickle` means the data is a pickled `dataframe` file, we will load it directly.",
    )
    valid_data_type = StrField(
        value="none",
        options=["dict", "dataframe", "parquet", "json", "none"],
        help="the type of valid data, `none` means no valid data",
    )
    test_data_type = StrField(
        value="none",
        options=["dict", "dataframe", "parquet", "json", "none"],
        help="the type of test data, `none` means no test data",
    )
    predict_data_type = StrField(
        value="none",
        options=["dict", "dataframe", "parquet", "parquet_list", "json", "none"],
        help="the type of predict data, `none` means no predict data",
    )
    online_data_type = StrField(
        value="none",
        options=["dict", "dataframe", "none"],
        help="the type of online data, `none` means no online data",
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


def yield_dataframe(origin, data_type, config: DefaultProcessorConfig):
    if data_type == "none":
        yield None
    elif data_type == "dataframe":
        yield origin
    elif data_type == "dict":
        yield pd.DataFrame(data=origin)
    elif data_type == "pickle":
        assert isinstance(origin, str)
        if config.data_root:
            origin = os.path.join(config.data_root, origin)
        yield pkl.load(open(origin, "rb"))
    elif data_type == "json":
        assert isinstance(origin, str)
        if config.data_root:
            origin = os.path.join(config.data_root, origin)
        yield pd.read_json(origin)
    elif data_type == "parquet":
        assert isinstance(origin, str)
        if config.data_root:
            origin = os.path.join(config.data_root, origin)
        yield pq.read_table(origin).to_pandas()
    elif data_type == "parquet_list":
        assert isinstance(origin, list)
        for path in origin:
            if config.data_root:
                path = os.path.join(config.data_root, path)
            yield pq.read_table(path).to_pandas()
    else:
        raise NotImplementedError


@register("processor", "default")
class DefaultProcessor(object):
    """docstring for IProcessor"""

    stage_data_set_map = {
        "train": "train_data_set",
        "predict": "predict_data_set",
        "online": "online_data_set",
    }

    def __init__(self, stage: str, config: DefaultProcessorConfig):
        super(DefaultProcessor, self).__init__()
        config_dict = config._to_dict()
        self.stage = stage
        self.config: DefaultProcessorConfig = config

        assert (
            (not self.config.meta_collection_on_train)
            or (not self.config.load_meta_on_start)
            or (not self.config.train_data_type == "none")
        )

        self.subprocessors = {}
        droped_subprocessors = []
        self.will_processed_data_set = set()
        for name in self.config.feed_order:
            subprocessor_config_dict = config_dict[f"@subprocessor@{name}"]
            if not subprocessor_config_dict[self.stage_data_set_map[stage]]:
                logger.info(f"Skip '{name}' ....")
                droped_subprocessors.append(name)
                continue
            self.will_processed_data_set.update(
                set(subprocessor_config_dict[self.stage_data_set_map[stage]])
            )
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
            self.subprocessors[name] = subprocessor
        for name in droped_subprocessors:
            self.config.feed_order.remove(name)

    def load_data(self, data: Dict, type_name: str) -> Iterator:
        """load data
        Returns:
            Iterable DataFrame
        """
        if type_name == "train":
            return yield_dataframe(
                data.get("train", {}), self.config.train_data_type, self.config
            )
        if type_name == "valid":
            return yield_dataframe(
                data.get("valid", {}), self.config.valid_data_type, self.config
            )
        if type_name == "test":
            return yield_dataframe(
                data.get("test", {}), self.config.test_data_type, self.config
            )
        if type_name == "predict":
            return yield_dataframe(
                data.get("predict", {}), self.config.predict_data_type, self.config
            )
        raise NotImplementedError

    def save(self, data: pd.DataFrame, type_name: str, i: int):
        """save data to self.config.processed_data_dir

        Args:
            data: should saved data

        Returns:
            None
        """
        os.makedirs(
            os.path.join(self.config.processed_data_dir, type_name), exist_ok=True
        )
        data_path = os.path.join(self.config.processed_data_dir, type_name, f"{i}.pkl")
        assert (
            i == 0
        ), f"Currently only support save one {type_name} data"  # TODO: support save multi data
        pkl.dump(data, open(data_path, "wb"))

    def process(self, data: Dict) -> Dict:
        """Process entry

        Args:
            data:
            >>> {
            >>>     "train": {training data....},
            >>>     "test": ..
            >>> }

        Returns:
            processed data
        """
        result = {}
        if not self.config.do_save:
            result = {key: [] for key in data}
        for type_name in ["train", "valid", "test", "predict"]:
            if type_name not in self.will_processed_data_set:
                continue
            for i, loaded_data in enumerate(self.load_data(data, type_name)):
                if loaded_data is None:
                    continue
                deliver_meta = (
                    self.config.meta_collection_on_train
                    and type_name == "train"
                    and i == 0
                )
                for name in self.config.feed_order:
                    logger.info(
                        f"Processing on {type_name} {i if i > 0 else ''}: {name}"
                    )
                    loaded_data = self.subprocessors[name].process(
                        data=loaded_data, deliver_meta=deliver_meta
                    )
                if not self.config.do_save:
                    result[type_name].append(loaded_data)
                else:
                    self.save(loaded_data, type_name, i)
        return result

    def online_process(self, data: pd.DataFrame):
        """online server process the data without save
        Args:
            data:
                the data to be processed

        Returns:
            processed data
        """
        if "online" not in self.will_processed_data_set:
            return data
        for name in self.config.feed_order:
            data = self.subprocessors[name].process(data=data, deliver_meta=False)
        return data
