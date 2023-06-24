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

from typing import Dict, List
from lightning.pytorch.callbacks import StochasticWeightAveraging
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, DictField, SubModules

@config_register("callback", 'weight_average')
@define
class StochasticWeightAveragingCallbackConfig(BaseConfig):
    name = NameField(value="weight_average", file=__file__, help="Implements the Stochastic Weight Averaging (SWA) Callback to average a model.")
    @define
    class Config:
        swa_lrs = AnyField(value=None, help="The SWA learning rate to use: float. Use this value for all parameter groups of the optimizer. List[float]. A list values for each parameter group of the optimizer.")
        swa_epoch_start = FloatField(value=0.8, checker=float_check(lower=0.0, upper=1.0), help="If provided as int, the procedure will start from the ``swa_epoch_start``-th epoch. If provided as float between 0 and 1, the procedure will start from ``int(swa_epoch_start * max_epochs)`` epoch")
        annealing_epochs = IntField(value=10, checker=int_check(lower=0), help="""number of epochs in the annealing phase (default: 10)""")
        annealing_strategy = StrField(value='cos', checker=str_check(options=['cos', 'linear'], additions=None), help="""Specifies the annealing strategy (default: "cos"): ``"cos"``. For cosine annealing. ``"linear"`` For linear annealing""")
        device = StrField(value=None, checker=str_check(options=['cpu', 'cuda'], additions=None), help="""if provided, the averaged model will be stored on the ``device``. When None is provided, it will infer the `device` from ``pl_module``. (default: ``"cpu"``)""")
    config = NestField(value=Config, converter=nest_converter)



@register("callback", 'weight_average')
class StochasticWeightAveragingCallback(object):
    """Average weight by config
    """

    def __init__(self, config: StochasticWeightAveragingCallbackConfig):
        super().__init__()
        self.config = config.to_dict()['config']

    def __call__(self, rt_config: Dict)->StochasticWeightAveraging:
        """return StochasticWeightAveraging object

        Args:
            rt_config: runtime config, include save_dir, and the checkpoint path name

        Returns: 
            StochasticWeightAveraging object

        """
        return StochasticWeightAveraging(**self.config)
