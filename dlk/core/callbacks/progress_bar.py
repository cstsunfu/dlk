# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Union
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar, ProgressBar
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, DictField, SubModules
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
import lightning.pytorch as pl

@config_register("callback", 'prograss_bar')
@define
class ProgressBarCallbackConfig(BaseConfig):
    name = NameField(value="prograss_bar", file=__file__, help="the prograss bar callback for lightning")
    @define
    class Config:
        type_bar = StrField(value='tqdm', checker=str_check(options=['tqdm', 'rich'], additions=None), help="""the type of the prograss bar, ``tqdm`` or ``rich``. Defaults to ``tqdm``.""")
        refresh_rate = IntField(value=1, checker=int_check(lower=0), help="Determines at which rate (in number of batches) the progress bars get updated. Set it to ``0`` to disable the display. Defaults to ``1``.")
        leave = BoolField(value=False, help="Leaves the finished progress bar in the terminal at the end of the epoch. Defaults to ``False``.")
        theme = StrField(value='default', checker=options(['default']), help="When the `type_bar` is `rich`. Contains styles used to stylize the progress bar. Currentlly only support `default`.")
        process_position = IntField(value=0, checker=int_check(), help="Set this to a value greater than ``0`` to offset the progress bars by this many lines. This is useful when you have progress bars defined elsewhere and want to show all of them together.")
    config = NestField(value=Config, converter=nest_converter)

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
        standard_metrics = {} # NOTE: remove the `v_num` get_standard_metrics(trainer)
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
        standard_metrics = {} # NOTE: remove the `v_num` get_standard_metrics(trainer)
        pbar_metrics = trainer.progress_bar_metrics
        duplicates = list(standard_metrics.keys() & pbar_metrics.keys())
        if duplicates:
            rank_zero_warn(
                f"The progress bar already tracks a metric with the name(s) '{', '.join(duplicates)}' and"
                f" `self.log('{duplicates[0]}', ..., prog_bar=True)` will overwrite this value. "
                " If this is undesired, change the name or override `get_metrics()` in the progress bar callback.",
            )

        return {**standard_metrics, **pbar_metrics}
        

@register("callback", 'prograss_bar')
class RichProgressBarCallback(object):
    """prograss bar callback for lightning
    """

    def __init__(self, config: ProgressBarCallbackConfig):
        super().__init__()
        self.config = config.to_dict()['config']

    def __call__(self, rt_config: Dict)->ProgressBar:
        """return LearningRateMonitor object

        Args:
            rt_config: runtime config, include save_dir, and the checkpoint path name

        Returns: 
            RichProgressBar object

        """
        if self.config['type_bar'] == 'tqdm':
            return NewTQDMProgressBar(refresh_rate=self.config['refresh_rate'], process_position=self.config['process_position'])
        else:
            assert self.config['type_bar'] == 'rich'
            theme = None
            assert self.config['theme'] == 'default'
            from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
            theme = RichProgressBarTheme()
            return NewRichProgressBar(refresh_rate=self.config['refresh_rate'], leave=self.config['leave'], theme=theme)
