# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

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
from lightning.pytorch.callbacks import StochasticWeightAveraging

from dlk.utils.register import register


@cregister("callback", "weight_average")
class StochasticWeightAveragingCallbackConfig(Base):
    """callback for the Stochastic Weight Averaging to average a model weight on the training process"""

    swa_lrs = AnyField(
        value=None,
        help="The SWA learning rate to use: float. Use this value for all parameter groups of the optimizer. List[float]. A list values for each parameter group of the optimizer.",
    )
    swa_epoch_start = FloatField(
        value=0.8,
        minimum=0.0,
        maximum=1.0,
        help="If provided as int, the procedure will start from the ``swa_epoch_start``-th epoch. If provided as float between 0 and 1, the procedure will start from ``int(swa_epoch_start * max_epochs)`` epoch",
    )
    annealing_epochs = IntField(
        value=10,
        minimum=0,
        help="""number of epochs in the annealing phase (default: 10)""",
    )
    annealing_strategy = StrField(
        value="cos",
        options=["cos", "linear"],
        help="""Specifies the annealing strategy (default: "cos"): ``"cos"``. For cosine annealing. ``"linear"`` For linear annealing""",
    )
    device = StrField(
        value="cpu",
        options=["cpu", "cuda"],
        help="""if provided, the averaged model will be stored on the ``device``. When None is provided, it will infer the `device` from ``pl_module``. (default: ``"cpu"``)""",
    )


@register("callback", "weight_average")
class StochasticWeightAveragingCallback(object):
    """Average weight by config"""

    def __init__(self, config: StochasticWeightAveragingCallbackConfig):
        super().__init__()
        self.config = config._to_dict(only_para=True)

    def __call__(self, rt_config: Dict) -> StochasticWeightAveraging:
        """return StochasticWeightAveraging object

        Args:
            rt_config: runtime config, include save_dir, and the checkpoint path name

        Returns:
            StochasticWeightAveraging object

        """
        return StochasticWeightAveraging(**self.config)
