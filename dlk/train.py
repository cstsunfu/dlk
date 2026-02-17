# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import copy
import gc
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

import dlk.adv_method
import dlk.callback
import dlk.data.data_collate
import dlk.data.datamodule
import dlk.data.dataset
import dlk.data.postprocessor
import dlk.data.processor
import dlk.data.subprocessor
import dlk.imodel
import dlk.initmethod
import dlk.loss
import dlk.nn
import dlk.optimizer
import dlk.scheduler
import dlk.trainer
from dlk.utils.io import open
from dlk.utils.logger import change_log_file, setup_logger
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


def setup_ray_logging(method="tune"):
    """
    Correctly removes the StreamHandler from the ROOT logger for non-zero ranks.
    """
    from ray import train, tune

    if method == "train":
        rank = train.get_context().get_world_rank()
    elif method == "tune":
        rank = tune.get_context().get_world_rank()
    else:
        raise ValueError("Unsupported method. Use 'train' or 'tune'.")

    if rank != 0:
        # 获取根 logger，而不是具名 logger
        root_logger = logging.getLogger()

        # 遍历根 logger 的 handlers
        for handler in root_logger.handlers[:]:
            # 移除 StreamHandler 来禁止控制台输出
            if isinstance(handler, logging.StreamHandler):
                root_logger.removeHandler(handler)
                # 打印一条调试信息到文件日志（如果已配置）
                root_logger.debug(
                    f"Rank {rank}: Removed StreamHandler to suppress console output."
                )
                break  # 假设只有一个 StreamHandler


class Train(object):
    """Trainer"""

    def __init__(
        self,
        config: Union[str, Dict],
        checkpoint: str = "",
        state_dict_only=True,
        strict=False,
        update_config: Union[Dict, None] = None,
    ):
        super(Train, self).__init__()
        config_dict = {}
        self.load_checkpoint_strict = strict
        if not isinstance(config, dict):
            with open(config, "r") as f:
                config_dict = hjson.load(f, object_pairs_hook=dict)
        else:
            config_dict = config

        self.ray_tune_report_stage = None

        self.checkpoint = checkpoint
        self.state_dict_only = state_dict_only
        self.config_dict = config_dict
        self.update_config = update_config

    def run(self, optuna_skip: Callable = lambda x: False):
        """run for all configs

        Returns:
            None

        """

        if not self.config_dict.get("@optuna", {}):
            self.default_run()
            return
        self.optuna_run(optuna_skip)

    def optuna_run(self, optuna_skip):
        configs = Parser(self.config_dict, update_config=self.update_config).parser(
            parser_ref=False
        )
        assert len(configs) == 1, "Currently only support one config for optuna."
        config = configs[0]
        import ray
        from ray import tune

        from dlk.ray_optuna import RayOptunaConfig, prepare_tune

        ray.init(ignore_reinit_error=True)
        optuna_config = RayOptunaConfig._from_dict(config.pop("@optuna"))
        self.ray_tune_report_stage = optuna_config.report_stage

        asha_scheduler, optuna_search, search_space = prepare_tune(optuna_config)

        def _trial(opt_paras):
            from lightning.pytorch.trainer import trainer as trainer_mod

            trainer_mod.Trainer._teardown = lambda self: None

            specific_keys = search_space.keys()
            config_name = []

            hyper_paras = {}
            for key in specific_keys:
                config_name.append(f"{key}={opt_paras[key]}")
                hyper_paras[key] = opt_paras[key]
            config_name = "/".join(config_name)
            cur_config = copy.deepcopy(config)
            cur_config["_G"].update(hyper_paras)
            parserd_cur_config = Parser(
                cur_config, update_config=self.update_config
            ).parser_init()[0]

            self.run_oneturn(parserd_cur_config, config_name, hyper_paras)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

        analysis = tune.run(
            _trial,
            config=search_space,
            num_samples=optuna_config.num_trials,
            search_alg=optuna_search,
            scheduler=asha_scheduler,
            resources_per_trial=optuna_config.resources_per_trial,
            verbose=optuna_config.verbose,
            resume=optuna_config.resume,
        )

        change_log_file(os.path.join(config.log_dir, "log.txt"))
        best_config = analysis.get_best_config(
            metric=optuna_config.metric, mode=optuna_config.mode
        )
        logger.info(
            f"Optuna tuning finished. Best config: {best_config}, best value: {analysis.best_result[optuna_config.metric]}"
        )
        ray.shutdown()

    def default_run(self):
        configs = Parser(
            self.config_dict, update_config=self.update_config
        ).parser_init()
        if self.checkpoint:
            assert (
                len(configs) == 1
            ), f"Reuse the checkpoint(checkpoint is not none), you must provide the (only one) config which generate the checkpoint."

        config_names = []
        hyper_parameters = []
        for i, possible_config in enumerate(configs):
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
                config_names.append("/".join(config_name))
                hyper_parameters.append(hyper_parameter)
            else:
                config_names.append(str(i))
                hyper_parameters.append({})
        logger.info(
            f"You have {len(config_names)} training config(s), they all will be run."
        )
        for i, (config, name, hyper_config) in enumerate(
            zip(configs, config_names, hyper_parameters)
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
        save_dir = os.path.join(config.log_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump({"@fit": config._to_dict()}, f, ensure_ascii=False, indent=4)
        change_log_file(os.path.join(save_dir, "log.txt"))

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
        imodel = self.get_imodel(config, data, hyper_config, name)

        # start training
        if "valid" in data or "train" in data:
            trainer.fit(model=imodel, datamodule=datamodule)
        if "test" in data:
            trainer.test(model=imodel, datamodule=datamodule)

    def get_data(self, config: DLKFitConfig):
        from datasets import load_from_disk

        data = {}
        for data_type in ["train", "valid", "test"]:
            data_path = os.path.join(config.processed_data_dir, data_type)
            if os.path.exists(data_path) and os.path.isdir(data_path):
                try:
                    data[data_type] = load_from_disk(data_path)
                except Exception as e:
                    logger.warning(f"Failed to load dataset from {data_path}: {e}")
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
                "ray_tune_report_stage": self.ray_tune_report_stage,
                "hp_metrics": config.hp_metrics,
                "hyper_config": hyper_config,
            },
        )
        return trainer

    def get_imodel(self, config: DLKFitConfig, data, hyper_config, name):
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
                "name": name,
                "log_dir": config.log_dir,
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

        imodel.train()
        return imodel
