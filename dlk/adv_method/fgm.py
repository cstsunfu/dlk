# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
import re
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
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

from dlk.utils.register import register

from . import AdvMethod, AdvMethodConfig

logger = logging.getLogger(__name__)


@cregister("adv_method", "fgm")
class FGMAdvMethodConfig(AdvMethodConfig):
    """
    FGM adversarial training method config
    https://arxiv.org/pdf/1706.06083.pdf
    """

    embedding_pattern = StrField(
        value="model.*embedding.*embedding",
        help="the pattern of embedding name that FGM effect on",
    )
    epsilon = FloatField(
        value=1.0,
        minimum=0.0,
        help="epsilon for FGM adversarial training",
    )


@register("adv_method", "fgm")
class FGMAdvMethod(AdvMethod):
    """FGM adversarial training method"""

    def __init__(self, model: nn.Module, config: FGMAdvMethodConfig):
        super().__init__(model, config)
        self.model = model
        self.config = config
        self.backup = {}
        self.adv_para_name = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.findall(self.config.embedding_pattern, name):
                self.adv_para_name.add(name)
        logger.info(
            f"There are {len(self.adv_para_name)} paras will be adversarial training."
        )
        logger.info(f"They are {self.adv_para_name}.")

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.adv_para_name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.config.alpha * param.grad / (norm + 1e-8)
                    with torch.no_grad():
                        param.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.adv_para_name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def training_step(self, imodel, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Performs a training step using the Fast Gradient Method (FGM).

        This method supports PyTorch Lightning's Automatic Mixed Precision (AMP) by
        utilizing `manual_backward` for gradient scaling and `optimizer.step()` for
        unscaling and updating weights.

        Args:
            imodel: The integrated model instance (LightningModule).
            batch: A dictionary containing the mini-batch inputs.
            batch_idx: The index of the current batch.

        Returns:
            Tuple[torch.Tensor, Dict]: A tuple containing the combined loss and the loss log.
        """
        optimizer = imodel.optimizers()
        rt_config = {
            "current_step": imodel.global_step,
            "current_epoch": imodel.current_epoch,
            "total_steps": imodel.num_training_steps,
            "total_epochs": imodel.num_training_epochs,
        }

        optimizer.zero_grad()

        # 1. Forward and backward for clean data
        result = imodel.model.training_step(batch)
        loss, loss_log = imodel.calc_loss(result, batch, rt_config=rt_config)
        imodel.manual_backward(loss / 2.0)

        # 2. Attack: Calculate perturbations using current gradients and apply to embeddings
        self.attack()

        # 3. Forward and backward for adversarial data
        result = imodel.model.training_step(batch)
        adv_loss, adv_loss_log = imodel.calc_loss(result, batch, rt_config=rt_config)
        imodel.manual_backward(adv_loss / 2.0)

        # 4. Restore clean parameters
        self.restore()

        # 5. Optimizer step: Lightning handles AMP unscaling internally
        imodel.clip_gradients(
            optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        optimizer.step()
        imodel.lr_schedulers().step()

        return (loss + adv_loss) / 2.0, loss_log
