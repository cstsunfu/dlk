import os
from typing import Dict, Union

import torch.nn as nn
from intc import (
    MISSING,
    AnyField,
    Base,
    BoolField,
    DictField,
    EnumField,
    FloatField,
    IntField,
    ListField,
    NestField,
    StrField,
    SubModule,
    cregister,
    dataclass,
)

from dlk.utils.import_module import import_module_dir
from dlk.utils.register import register


@cregister("optuna")
class RayOptunaConfig(Base):
    """the base loss config"""

    study_name = StrField(value=MISSING, help="The name of the Optuna study.")
    search_space = DictField(
        value={},
        help="""
        The search space for hyperparameters, like
        {   
            'lr': ['loguniform', [1e-4, 1e-1]], 
            'batch_size': ['choice', [16, 32, 64, 128]
        }
        Saample methods:
        - 'uniform': uniform distribution
        - 'loguniform': log-uniform distribution
        - 'choice': choose from a list of values
        - 'randint': random integer from a range
        - 'lograndint': random integer from a log-uniform distribution
        - 'randn': random normal distribution
        """,
    )
    num_trials = IntField(
        value=10, help="Number of times to sample from the hyperparameter space."
    )
    report_stage = StrField(
        value="train-epoch-end",
        help="The stage at which to report the results.",
        options=["train-epoch-end", "validation-epoch-end", "test-epoch-end"],
    )
    max_t = IntField(
        value=10**8,
        help="The maximum number of training iterations for each trial(default stop by trainer).",
    )
    metric = StrField(value=MISSING, help="The metric to optimize, e.g., 'accuracy'.")
    mode = StrField(
        value="max",
        help="The optimization mode, either 'min' or 'max'.",
        options=["min", "max"],
    )
    grace_period = IntField(
        value=1,
        help="The grace period for early stopping, i.e., the number of iterations to wait before considering a trial for early stopping.",
    )
    reduction_factor = FloatField(
        value=1.5,
        help="The reduction factor for the ASHA scheduler, i.e., how many trials( 1/reduction_factor * 100%) to keep after each iteration.",
    )

    time_attr = StrField(
        value="epoch", help="The attribute to use for time tracking in the scheduler."
    )
    resources_per_trial = DictField(
        value=MISSING,
        help="The resources allocated per trial, e.g., {'cpu': 1, 'gpu': 0}.",
    )
    resume = EnumField(
        value="AUTO",
        options=["AUTO", True, False],
        help="Whether to resume from a previous run. 'AUTO' will automatically detect if a previous run exists.",
    )
    verbose = IntField(
        value=1,
        help="Verbosity level for the tuning process.  0 = silent, 1 = default, 2 = verbose",
    )


def prepare_tune(config: RayOptunaConfig):
    import optuna
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    function_map = {
        "uniform": tune.uniform,
        "loguniform": tune.loguniform,
        "choice": tune.choice,
        "randint": tune.randint,
        "lograndint": tune.lograndint,
        "randn": tune.randn,
    }
    asha_scheduler = ASHAScheduler(
        metric=config.metric,
        mode=config.mode,
        max_t=config.max_t,
        grace_period=config.grace_period,
        reduction_factor=config.reduction_factor,
        time_attr=config.time_attr,
    )

    rdb_storage = optuna.storages.RDBStorage(url=f"sqlite:///{config.study_name}.db")
    optuna_search = OptunaSearch(
        metric="mean_accuracy",
        mode="max",
        study_name=config.study_name,
        storage=rdb_storage,
    )

    search_space = {}
    for key, value in config.search_space.items():
        if isinstance(value, str):
            assert value.startswith(
                "lambda "
            ), f"Invalid search space format for {key}: {value}. 'lambda c: c.config.a + 1' for dependent config."
            search_space[key] = tune.sample_from(eval(value))
        elif isinstance(value, list) and len(value) == 2:
            method, args = value
            if method in function_map:
                search_space[key] = function_map[method](*args)
            else:
                raise ValueError(f"Unknown sampling method: {method}")
        else:
            raise ValueError(f"Invalid search space format for {key}: {value}")
    return asha_scheduler, optuna_search, search_space
