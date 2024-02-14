# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

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
from lightning.pytorch.callbacks import LearningRateMonitor

from dlk.utils.register import register


@cregister("callback", "lr_monitor")
class LearningRateMonitorCallbackConfig(Base):
    """callback for monitor the learning rate of the optimizer"""

    logging_interval = StrField(
        value=None,
        options=["step", "epoch", None],
        help="""set to ``'epoch'`` or ``'step'`` to log ``lr`` of all optimizers at the same interval, set to ``None`` to log at individual interval according to the ``interval`` key of each scheduler. Defaults to ``None``.""",
    )
    log_momentum = BoolField(
        value=False,
        help="option to also log the momentum values of the optimizer, if the optimizer has the ``momentum`` or ``betas`` attribute. Defaults to ``False``.",
    )


@register("callback", "lr_monitor")
class LearningRateMonitorCallback(object):
    """Monitor the learning rate"""

    def __init__(self, config: LearningRateMonitorCallbackConfig):
        super().__init__()
        self.config = config._to_dict(only_para=True)

    def __call__(self, rt_config: Dict) -> LearningRateMonitor:
        """return LearningRateMonitor object

        Args:
            rt_config: runtime config, include save_dir, and the checkpoint path name

        Returns:
            LearningRateMonitor object

        """
        return LearningRateMonitor(**self.config)
