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

"""schedulers"""
import importlib
import os
from dlk.utils.register import Register
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math


scheduler_config_register = Register("Schedule config register.")
scheduler_register = Register("Schedule register.")


class BaseScheduler(object):
    """interface for Schedule"""

    def get_scheduler(self)->LambdaLR:
        """return the initialized scheduler

        Returns: 
            Schedule

        """
        raise NotImplementedError

    def __call__(self):
        """the same as self.get_scheduler()
        """
        return self.get_scheduler()


def import_schedulers(schedulers_dir, namespace):
    for file in os.listdir(schedulers_dir):
        path = os.path.join(schedulers_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            scheduler_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + scheduler_name)


# automatically import any Python files in the schedulers directory
schedulers_dir = os.path.dirname(__file__)
import_schedulers(schedulers_dir, "dlk.core.schedulers")
