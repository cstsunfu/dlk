# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict

import torch
import torch.nn as nn

from dlk.utils.import_module import import_module_dir


class AdvMethod(object):
    """Save fgm decided by config"""

    def __init__(self, model: nn.Module, config):
        super().__init__()

    def training_step(self, imodel, batch: Dict[str, torch.Tensor], batch_idx: int):
        """do training_step on a mini batch

        Args:
            imodel: imodel instance
            batch: a mini batch inputs
            batch_idx: the index(dataloader) of the mini batch

        Returns:
            the outputs

        """
        raise NotImplementedError


# automatically import any Python files in the models directory
adv_method_dir = os.path.dirname(__file__)
import_module_dir(adv_method_dir, "dlk.adv_method")
