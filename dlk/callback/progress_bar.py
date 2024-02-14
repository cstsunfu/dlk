# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Union

import lightning.pytorch as pl
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
from lightning.pytorch.callbacks import ProgressBar, RichProgressBar, TQDMProgressBar
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

from dlk.utils.register import register


@cregister("callback", "prograss_bar")
class ProgressBarCallbackConfig(Base):
    """callback for the prograss bar"""

    type_bar = StrField(
        value="tqdm",
        options=["tqdm", "rich"],
        help="""the type of the prograss bar, ``tqdm`` or ``rich``. Defaults to ``tqdm``.""",
    )
    refresh_rate = IntField(
        value=1,
        minimum=0,
        help="Determines at which rate (in number of batches) the progress bars get updated. Set it to ``0`` to disable the display. Defaults to ``1``.",
    )
    leave = BoolField(
        value=False,
        help="Leaves the finished progress bar in the terminal at the end of the epoch. Defaults to ``False``.",
    )
    theme = StrField(
        value="default",
        options=["default"],
        help="When the `type_bar` is `rich`. Contains styles used to stylize the progress bar. Currently only support `default`.",
    )
    process_position = IntField(
        value=0,
        help="Set this to a value greater than ``0`` to offset the progress bars by this many lines. This is useful when you have progress bars defined elsewhere and want to show all of them together.",
    )


class NewRichProgressBar(RichProgressBar):
    """docstring for  ProgressBar"""

    def __init__(self, **config):
        super(NewRichProgressBar, self).__init__(**config)

    def get_metrics(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> Dict[str, Union[int, str, float, Dict[str, float]]]:
        r"""Combines progress bar metrics collected from the trainer with standard metrics from
        get_standard_metrics. Implement this to override the items displayed in the progress bar.

        Here is an example of how to override the defaults:

        .. code-block:: python

            def get_metrics(self, trainer, model):
                # don't show the version number
                items = super().get_metrics(trainer, model)
                items.pop("v_num", None)
                return items

        Return:
            Dictionary with the items to be displayed in the progress bar.
        """
        standard_metrics = {}  # NOTE: remove the `v_num` get_standard_metrics(trainer)
        pbar_metrics = trainer.progress_bar_metrics
        duplicates = list(standard_metrics.keys() & pbar_metrics.keys())
        if duplicates:
            rank_zero_warn(
                f"The progress bar already tracks a metric with the name(s) '{', '.join(duplicates)}' and"
                f" `self.log('{duplicates[0]}', ..., prog_bar=True)` will overwrite this value. "
                " If this is undesired, change the name or override `get_metrics()` in the progress bar callback.",
            )

        return {**standard_metrics, **pbar_metrics}


class NewTQDMProgressBar(TQDMProgressBar):
    """docstring for  ProgressBar"""

    def __init__(self, **config):
        super(NewTQDMProgressBar, self).__init__(**config)

    def get_metrics(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> Dict[str, Union[int, str, float, Dict[str, float]]]:
        r"""Combines progress bar metrics collected from the trainer with standard metrics from
        get_standard_metrics. Implement this to override the items displayed in the progress bar.

        Here is an example of how to override the defaults:

        .. code-block:: python

            def get_metrics(self, trainer, model):
                # don't show the version number
                items = super().get_metrics(trainer, model)
                items.pop("v_num", None)
                return items

        Return:
            Dictionary with the items to be displayed in the progress bar.
        """
        standard_metrics = {}  # NOTE: remove the `v_num` get_standard_metrics(trainer)
        pbar_metrics = trainer.progress_bar_metrics
        duplicates = list(standard_metrics.keys() & pbar_metrics.keys())
        if duplicates:
            rank_zero_warn(
                f"The progress bar already tracks a metric with the name(s) '{', '.join(duplicates)}' and"
                f" `self.log('{duplicates[0]}', ..., prog_bar=True)` will overwrite this value. "
                " If this is undesired, change the name or override `get_metrics()` in the progress bar callback.",
            )

        return {**standard_metrics, **pbar_metrics}


@register("callback", "prograss_bar")
class RichProgressBarCallback(object):
    """prograss bar callback for lightning"""

    def __init__(self, config: ProgressBarCallbackConfig):
        super().__init__()
        self.config = config

    def __call__(self, rt_config: Dict) -> ProgressBar:
        """return LearningRateMonitor object

        Args:
            rt_config: runtime config, include save_dir, and the checkpoint path name

        Returns:
            RichProgressBar object

        """
        if self.config.type_bar == "tqdm":
            return NewTQDMProgressBar(
                refresh_rate=self.config.refresh_rate,
                process_position=self.config.process_position,
            )
        else:
            assert self.config.type_bar == "rich"
            theme = None
            assert self.config.theme == "default"
            from lightning.pytorch.callbacks.progress.rich_progress import (
                RichProgressBarTheme,
            )

            theme = RichProgressBarTheme()
            return NewRichProgressBar(
                refresh_rate=self.config.refresh_rate,
                leave=self.config.leave,
                theme=theme,
            )
