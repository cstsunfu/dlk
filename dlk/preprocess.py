# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Dict, Union

import hjson
import pandas as pd
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
    Parser,
    StrField,
    SubModule,
    cregister,
    init_config,
)

from dlk.utils.io import open
from dlk.utils.register import register, register_module_name


class PreProcessor(object):
    """Processor"""

    def __init__(self, config: Union[str, dict], stage: str = "train"):
        super(PreProcessor, self).__init__()
        config_dict = {}
        if not isinstance(config, dict):
            with open(config, "r") as f:
                config_dict = hjson.load(f, object_pairs_hook=dict)
        else:
            config_dict = config
        prepro_config = self.get_config(config_dict)
        self.init(prepro_config, stage)

    def init(self, config: Base, stage: str):
        """init the model and trainer
        Args:
            config: the preprocess config
            stage: train/predict/online

        Returns: None
        """
        # set processor
        process_name = register_module_name(config._module_name)
        processor_ins = register.get("processor", process_name)(
            config=config, stage=stage
        )
        if stage in ["train", "predict"]:
            self.processor = processor_ins.process
        else:
            assert stage == "online", f"stage {stage} is not supported"
            self.processor = processor_ins.online_process

    def get_config(self, config_dict):
        """get the predict config

        Args:
            config: the init config

        Returns:
            DLKPreProConfig, config_name_str
        """
        configs = Parser(config_dict).parser_init()
        assert len(configs) == 1, f"You should not use '_search' for preprocess"

        prepro_config = configs[0]["@processor"]
        return prepro_config

    def fit(self, data: Union[Dict[str, Any], pd.DataFrame]):
        """Process the data and return the processed data

        Args:
            data: {"train": train DataFrame, 'valid': valid DataFrame} if stage is not online else DataFrame

        Returns:
            processed data

        """
        return self.processor(data)
