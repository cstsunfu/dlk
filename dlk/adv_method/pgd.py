# Copyright cstsunfu.
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


@cregister("adv_method", "pgd")
class PGDAdvMethodConfig(AdvMethodConfig):
    """
    PGD adversarial training method
    """

    embedding_pattern = StrField(
        value="model.*embedding.*embedding",
        help="the pattern of embedding name that PGD effect on",
    )
    epsilon = FloatField(
        value=1.0,
        minimum=0.0,
        help="epsilon for PGD adversarial training",
    )
    alpha = FloatField(
        value=0.3,
        minimum=0.0,
        help="alpha for PGD adversarial training",
    )
    adv_k = IntField(value=1, minimum=1, help="PGD adversarial training times")


@register("adv_method", "pgd")
class PGDAdvMethod(AdvMethod):
    """PGD adversarial training method"""

    def __init__(self, model: nn.Module, config: PGDAdvMethodConfig):
        super().__init__(model, config)
        self.model = model
        self.config = config
        self.emb_backup = {}
        self.grad_backup = {}
        self.adv_para_name = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad and re.findall(self.config.embedding_pattern, name):
                self.adv_para_name.add(name)
        logger.info(
            f"There are {len(self.adv_para_name)} paras will be adversarial training."
        )
        logger.info(f"They are {self.adv_para_name}.")

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.adv_para_name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                    continue
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.config.alpha * param.grad / (norm + 1e-8)
                    with torch.no_grad():
                        param.add_(r_at)
                    param.data = self.project(name, param.data, self.config.epsilon)

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.adv_para_name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.grad_backup:
                param.grad = param.grad + self.grad_backup[name]

    def training_step(self, imodel, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Performs a training step using Projected Gradient Descent (PGD).

        Args:
            imodel: The integrated model instance (LightningModule).
            batch: A dictionary containing the mini-batch inputs.
            batch_idx: The index of the current batch.

        Returns:
            Tuple[torch.Tensor, Dict]: A tuple containing the mean loss and the loss log.
        """
        optimizer = imodel.optimizers()
        rt_config = {
            "current_step": imodel.global_step,
            "current_epoch": imodel.current_epoch,
            "total_steps": imodel.num_training_steps,
            "total_epochs": imodel.num_training_epochs,
        }

        # 1. Backup clean embeddings
        self.attack(is_first_attack=True)

        # 2. Iteratively find perturbations
        for t in range(self.config.adv_k):
            optimizer.zero_grad()
            result = imodel.model.training_step(batch)
            loss, _ = imodel.calc_loss(result, batch, rt_config=rt_config)
            imodel.manual_backward(loss)
            self.attack(is_first_attack=False)

        # Clear gradients used for finding perturbations
        optimizer.zero_grad()

        # 3. Compute Adversarial Loss
        result = imodel.model.training_step(batch)
        adv_loss, adv_loss_log = imodel.calc_loss(result, batch, rt_config=rt_config)
        imodel.manual_backward(adv_loss / 2.0)

        # 4. Restore clean parameters
        self.restore()

        # 5. Compute Clean Loss
        result = imodel.model.training_step(batch)
        clean_loss, _ = imodel.calc_loss(result, batch, rt_config=rt_config)
        imodel.manual_backward(clean_loss / 2.0)

        # 6. Update parameters
        imodel.clip_gradients(
            optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        optimizer.step()
        imodel.lr_schedulers().step()

        return (clean_loss + adv_loss) / 2.0, adv_loss_log
