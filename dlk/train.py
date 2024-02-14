# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import pickle as pkl
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
from intc.utils import fix_trace

from dlk.utils.io import open
from dlk.utils.register import register, register_module_name

logger = logging.getLogger(__name__)


@cregister("fit")
class DLKFitConfig(Base):
    """"""

    specific = DictField(
        value={},
        help="""
        when we `_search` parameters we will save different config to different subdirectory. 
        it's a dict the pair is `{focus key: simplified key}`,
        `forcus key` is the path to focus key, like `@optimizer.lr`. 
        `simplified key` is the simplified name, like we can the simplify `@optimizer.lr` as `lr`. 
        if we do not provide the specific keys, we will just use 1, 2...n as the subdirectory name.
        """,
    )
    hp_metrics = StrField(
        value=None, additions=[None], help="the target metrics for logging"
    )
    log_dir = StrField(value="logs", help="the save dir of the config and logs")
    processed_data_dir = StrField(value=MISSING, help="the processed data path")
    submodule = SubModule(value={}, help="the submodule config")


class Train(object):
    """Trainer"""

    def __init__(
        self,
        config: Union[str, Dict],
        checkpoint: str = "",
        state_dict_only=True,
        strict=False,
    ):
        super(Train, self).__init__()
        config_dict = {}
        self.load_checkpoint_strict = strict
        if not isinstance(config, dict):
            with open(config, "r") as f:
                config_dict = hjson.load(f, object_pairs_hook=dict)
        else:
            config_dict = config

        self.checkpoint = checkpoint
        self.state_dict_only = state_dict_only
        self.configs = Parser(config_dict).parser_init()
        if self.checkpoint:
            assert (
                len(self.configs) == 1
            ), f"Reuse the checkpoint(checkpoint is not none), you must provide the (only one) config which generate the checkpoint."

        self.config_names = []
        self.hyper_parameters = []
        for i, possible_config in enumerate(self.configs):
            train_config = possible_config["@fit"]._to_dict()
            specific = train_config.get("specific", {})
            if specific:
                config_name = []
                hyper_parameter = {}
                for source, to in specific.items():
                    config_point = train_config
                    trace = fix_trace(source, train_config).split(".")
                    for t in trace:
                        config_point = config_point[t]
                    config_name.append(f"{to}={str(config_point)}")
                    hyper_parameter[to] = config_point
                self.config_names.append("/".join(config_name))
                self.hyper_parameters.append(hyper_parameter)
            else:
                self.config_names.append(str(i))
                self.hyper_parameters.append({})

    def run(self):
        """run for all configs

        Returns:
            None

        """
        logger.info(
            f"You have {len(self.config_names)} training config(s), they all will be run."
        )
        for i, (config, name, hyper_config) in enumerate(
            zip(self.configs, self.config_names, self.hyper_parameters)
        ):
            logger.info(f"Runing the {i}th {name}...")
            self.run_oneturn(config, name, hyper_config)

    def dump_config(self, config: DLKFitConfig, name: str):
        """dump the config and change the log file path to log_dir+name

        Args:
            config: the DLKFitConfig
            name: specific config name

        Returns:
            None

        """
        log_path = os.path.join(config.log_dir, name, "log.txt")
        with open(os.path.join(config.log_dir, name, "config.json"), "w") as f:
            json.dump({"@fit": config._to_dict()}, f, ensure_ascii=False, indent=4)

    def run_oneturn(self, base_config, name, hyper_config):
        """run this config

        Args:
            base_config: Base({"@fit": '...'})
            name: config name
            hyper_config: the hyper parameters

        Returns:
            None

        """

        config: DLKFitConfig = base_config["@fit"]
        # save configure
        self.dump_config(config, name)

        # set trainer
        trainer = self.get_trainer(config, name, hyper_config)

        # register devices for valid repeat
        # set datamodule
        datamodule, data = self.get_datamodule(config, world_size=trainer.world_size)

        # init imodel and inject the origin test and valid data
        imodel = self.get_imodel(config, data, hyper_config)

        # start training
        if "valid" in data or "train" in data:
            trainer.fit(model=imodel, datamodule=datamodule)
        if "test" in data:
            trainer.test(model=imodel, datamodule=datamodule)

    def get_data(self, config: DLKFitConfig):
        """get the data decided by config

        Returns:
            loaded all the processed data
        """
        # NOTE: currently only support load one data for each type
        # TODO: support load multi data for each type
        data = {}
        for data_type in ["train", "valid", "test"]:
            data_path = os.path.join(config.processed_data_dir, data_type, "0.pkl")
            if os.path.exists(data_path):
                with open(data_path, "rb") as f:
                    data[data_type] = pkl.load(f)
        return data

    def get_datamodule(self, config: DLKFitConfig, world_size):
        """get the datamodule decided by config, and fit the data to datamodule

        Args:
            config: DLKFitConfig
            devices: when the devices >1 and repeat_for_valid is True, we will repeat the

        Returns:
            datamodule

        """
        data = self.get_data(config)
        datamodule_configs = config._get_modules("datamodule")
        assert len(datamodule_configs) == 1, "Currently only support one datamodule"
        data_module_config = datamodule_configs[0]
        data_module_name = register_module_name(data_module_config._module_name)
        datamodule = register.get("datamodule", data_module_name)(
            data_module_config, data, {"world_size": world_size}
        )
        return datamodule, data

    def get_trainer(self, config: DLKFitConfig, name, hyper_config):
        """get the train/predict manager decided by config

        Args:
            config: DLKFitConfig
            name: the predict progress name
            hyper_config: the hyper parameters

        Returns:
            trainer

        """
        trainer_configs = config._get_modules("trainer")
        assert len(trainer_configs) == 1, "Currently only support one trainer"
        trainer_config = trainer_configs[0]
        trainer_name = register_module_name(trainer_config._module_name)
        trainer = register.get("trainer", trainer_name)(
            trainer_config,
            rt_config={
                "log_dir": config.log_dir,
                "name": name,
                "hp_metrics": config.hp_metrics,
                "hyper_config": hyper_config,
            },
        )
        return trainer

    def get_imodel(self, config: DLKFitConfig, data, hyper_config):
        """get the imodel decided by config, and inject the origin test and valid data

        Args:
            config: DLKFitConfig
            data: {"train": '..', 'valid': '..', ..}

        Returns:
            imodel

        """
        imodel_configs = config._get_modules("imodel")
        assert len(imodel_configs) == 1, "Currently only support one imodel"
        imodel_config = imodel_configs[0]
        imodel_name = register_module_name(imodel_config._module_name)
        imodel = register.get("imodel", imodel_name)(
            imodel_config,
            rt_config={
                "hp_metrics": config.hp_metrics,
                "hyper_config": hyper_config,
            },
        )
        if self.checkpoint:
            logger.info(f"reuse the checkpoint at {self.checkpoint}")
            if self.state_dict_only:
                with open(self.checkpoint, mode="rb") as f:
                    state_dict = torch.load(self.checkpoint)["state_dict"]
                imodel.load_state_dict(state_dict, strict=self.load_checkpoint_strict)
            else:
                raise NotImplementedError(
                    f"Currently not implement the load_from_checkpoint, only support load the `state_dict`"
                )
                # imodel.load_from_checkpoint(self.checkpoint)
        if "valid" in data:
            imodel._origin_data["valid"] = data["valid"]

        if "test" in data:
            imodel._origin_data["test"] = data["test"]

        return imodel
