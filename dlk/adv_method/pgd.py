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

from . import AdvMethod

logger = logging.getLogger(__name__)


@cregister("adv_method", "pgd")
class PGDAdvMethodConfig(Base):
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
    adv_k = IntField(value=3, minimum=1, help="PGD adversarial training times")


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
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.config.alpha * param.grad / norm
                    param.data.add_(r_at)
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
            if param.requires_grad:
                param.grad = self.grad_backup[name]

    def training_step(self, imodel, batch: Dict[str, torch.Tensor], batch_idx: int):
        """do training_step on a mini batch

        Args:
            imodel: imodel instance
            batch: a mini batch inputs
            batch_idx: the index(dataloader) of the mini batch

        Returns:
            the outputs

        """
        optimizer = imodel.optimizers()
        rt_config = {
            "current_step": imodel.global_step,
            "current_epoch": imodel.current_epoch,
            "total_steps": imodel.num_training_steps,
            "total_epochs": imodel.num_training_epochs,
        }
        optimizer.zero_grad()
        seed = random.randint(0, int(4e9))  # 4e9 < 2e32 - 1
        torch.manual_seed(seed)  # NOTE: should fix manual seed for every forward
        np.random.seed(seed)
        result = imodel.model.training_step(batch)
        loss, loss_log = imodel.calc_loss(result, batch, rt_config=rt_config)
        imodel.manual_backward(loss)

        self.backup_grad()
        for t in range(self.config.adv_k):
            self.attack(is_first_attack=(t == 0))

            if t != self.config.adv_k - 1:
                optimizer.zero_grad()
            else:
                self.restore_grad()
            result = imodel.model.training_step(batch)
            loss, loss_log = imodel.calc_loss(result, batch, rt_config=rt_config)
            imodel.manual_backward(loss)
        self.restore()
        optimizer.step()

        schedule = imodel.lr_schedulers()
        schedule.step()
        return loss, loss_log
