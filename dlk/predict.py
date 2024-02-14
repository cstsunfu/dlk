# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import logging
import os
import pickle as pkl
import uuid
from typing import Any, Callable, Dict, List, Union

import hjson
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

from dlk.train import DLKFitConfig
from dlk.utils.io import open
from dlk.utils.register import register, register_module_name

logger = logging.getLogger(__name__)


class Predict(object):
    """Predict

    Config Example:
    """

    def __init__(self, config: Union[str, dict], checkpoint: str):
        super(Predict, self).__init__()
        config_dict = {}
        self.online = False
        if not isinstance(config, dict):
            with open(config, "r") as f:
                config_dict = hjson.load(f, object_pairs_hook=dict)
        else:
            config_dict = config
        if isinstance(checkpoint, str):
            with open(checkpoint, "rb") as f:
                self.checkpoint = torch.load(f, map_location=torch.device("cpu"))
        else:
            self.checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
        dlk_config, name_str = self.get_config(config_dict)
        self.dlk_config = dlk_config
        self.init(dlk_config, name_str)

    def get_config(self, config_dict):
        """get the predict config

        Args:
            config: the init config

        Returns:
            DLKFitConfig, config_name_str
        """
        configs = Parser(config_dict).parser_init()
        assert len(configs) == 1, f"You should not use '_search' for predict/online"

        fit_config: DLKFitConfig = configs[0]["@fit"]

        config_name_str = "predict"
        return fit_config, config_name_str

    def init(self, config, name):
        """init the model and trainer
        Args:
            config: the config
            name: the name of the config

        Returns: None
        """
        # set trainer
        self.trainer = self.get_trainer(config, name)

        # init imodel and inject the origin test and valid data
        self.imodel = self.get_imodel(config)

    def predict(self, data=None, save_condition=False):
        """init the model, datamodule, manager then predict the predict_dataloader

        Args:
            data: if provide will not load from data_path

        Returns:
            None

        """
        # set datamodule
        datamodule, data = self.get_datamodule(
            self.dlk_config, data, world_size=self.trainer.world_size
        )

        # start predict
        with torch.no_grad():
            predict_result = self.trainer.predict(
                model=self.imodel, datamodule=datamodule
            )
        return self.imodel.postprocessor(
            stage="predict",
            list_batch_outputs=predict_result,
            origin_data=data["predict"],
            rt_config={},
            save_condition=save_condition,
        )

    def get_data(self, config):
        """get the data decided by config

        Args:
            config: {"config": {"data_path": '..'}}

        Returns:
            loaded data

        """
        data = {}
        for data_type in ["predict"]:
            data_path = os.path.join(config.processed_data_dir, data_type, "0.pkl")
            if os.path.exists(data_path):
                with open(data_path, "rb") as f:
                    data[data_type] = pkl.load(f)
        return data

    def get_datamodule(self, config, data, world_size):
        """get the datamodule decided by config, and fit the data to datamodule

        Args:
            config: {"task": {"datamodule": '..'}}
            data: {"train": '..', 'valid': '..', ..}

        Returns:
            datamodule

        """
        if not data and not self.online:
            data = self.get_data(config)
        datamodule_configs = config._get_modules("datamodule")
        assert len(datamodule_configs) == 1, "Currently only support one datamodule"
        data_module_config = datamodule_configs[0]
        data_module_name = register_module_name(data_module_config._module_name)
        datamodule = register.get("datamodule", data_module_name)(
            data_module_config, data, {"world_size": world_size}
        )
        return datamodule, data

    def get_trainer(self, config: DLKFitConfig, name):
        """get the train/predict manager decided by config

        Args:
            config: DLKFitConfig
            name: the predict progress name

        Returns:
            trainer

        """
        trainer_configs = config._get_modules("trainer")
        assert len(trainer_configs) == 1, "Currently only support one trainer"
        trainer_config = trainer_configs[0]
        trainer_name = register_module_name(trainer_config._module_name)
        trainer = register.get("trainer", trainer_name)(
            trainer_config, rt_config={"log_dir": config.log_dir, "name": name}
        )
        return trainer

    def get_imodel(self, config):
        """get the imodel decided by config

        Args:

        Returns:
            imodel

        """
        imodel_configs = config._get_modules("imodel")
        assert (
            len(imodel_configs) == 1
        ), f"Currently only support one imodel, {imodel_configs}"
        imodel_config = imodel_configs[0]
        imodel_name = register_module_name(imodel_config._module_name)
        imodel = register.get("imodel", imodel_name)(imodel_config, checkpoint=True)
        if self.checkpoint:
            imodel.load_state_dict(self.checkpoint["state_dict"], strict=True)
        imodel.eval()
        return imodel
