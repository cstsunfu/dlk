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

from dlk.data.subprocessor.fast_tokenizer import FastTokenizer, FastTokenizerConfig
from dlk.data.subprocessor.token2id import Token2ID, Token2IDConfig
from dlk.data.subprocessor.token_gather import TokenGather, TokenGatherConfig
from dlk.utils.register import register

logger = logging.getLogger(__name__)


@cregister("processor", "my")
class DefaultProcessorConfig(Base):
    """the default processor"""

    processed_data_dir = StrField(
        value="data/processed_data",
        help="the save dir of the processor, not effective by `data_root`.",
    )
    meta_dir = StrField(
        value="data/meta_data",
        help="the save dir of the meta info, not effective by `data_root`.",
    )
    tokenizer_path = StrField(
        value=MISSING,
        help="the config path for the tokenizer, this is not effected by `meta_dir`",
    )


@register("processor", "my")
class DefaultProcessor(object):
    """docstring for IProcessor"""

    stage_data_set_map = {
        "train": "train_data_set",
        "online": "online_data_set",
    }

    def __init__(self, stage: str, config: DefaultProcessorConfig):
        super(DefaultProcessor, self).__init__()
        config_dict = config._to_dict()
        self.stage = stage
        assert (
            stage in self.stage_data_set_map
        ), f"stage {stage} not supported for this example, the @processor@default support more"
        self.config: DefaultProcessorConfig = config
        self.tokenizer = FastTokenizer(  # for tokenize the data
            stage=stage,
            config=FastTokenizerConfig._from_dict(
                {"tokenizer_path": config.tokenizer_path}
            ),
            meta_dir=config.meta_dir,
        )
        self.label_gather = TokenGather(  # gather all the labels
            stage=stage,
            config=TokenGatherConfig._from_dict(
                {
                    "gather_columns": ["labels"],
                    "token_vocab": "label_vocab.json",
                    "unk": "",  # do not add unk and pad
                    "pad": "",
                }
            ),
            meta_dir=config.meta_dir,
        )
        self.label2id = Token2ID(  # convert the labels to label_ids
            stage=stage,
            config=Token2IDConfig._from_dict(
                {
                    "input_map": {"tokens": "labels"},
                    "output_map": {"token_ids": "label_ids"},
                    "vocab": "label_vocab.json",  # use the vocab from TokenGather
                }
            ),
            meta_dir=config.meta_dir,
        )

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
        assert i == 0, f"Currently only support save one {type_name} data"
        pkl.dump(data, open(data_path, "wb"))

    def process(self, data: Dict) -> Dict:
        """Process entry for train stage

        Args:
            data:
            >>> {
            >>>     "train": {training data....},
            >>>     "valid": ..
            >>> }

        Returns:
            processed data
        """
        train_data: pd.DataFrame = data["train"]
        valid_data: pd.DataFrame = data["valid"]
        train_data = self.tokenizer.process(train_data, deliver_meta=False)
        valid_data = self.tokenizer.process(valid_data, deliver_meta=False)
        self.label_gather.process(
            train_data, deliver_meta=True
        )  # will save the label vocab to `os.path.join(self.config.meta_dir, "label_vocab.json")`
        train_data = self.label2id.process(train_data, deliver_meta=False)
        valid_data = self.label2id.process(valid_data, deliver_meta=False)

        self.save(train_data, "train", 0)  # save the data
        self.save(valid_data, "valid", 0)

    def online_process(self, data: pd.DataFrame):
        """online server process the data without save, for online stage
        Args:
            data:
                the data to be processed

        Returns:
            processed data
        """
        data["label_ids"] = [[0]] * len(
            data
        )  # for online stage, just need to add a fake label_ids, just for donot consider the different of train and online for next step, or you can skip this step
        return self.tokenizer.process(
            data, deliver_meta=False
        )  # for online stage just need tokenized the data
