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


@cregister("adv_method", "free_lb")
class FreeLBAdvMethodConfig(AdvMethodConfig):
    """
    ENHANCED ADVERSARIAL TRAINING FOR NATURAL LANGUAGE UNDERSTANDING
    https://arxiv.org/pdf/1909.11764.pdf
    """

    embedding_pattern = StrField(
        value="model.*embedding.*embedding",
        help="the pattern of embedding name that FreeLB effect on",
    )
    epsilon = FloatField(
        value=1.0,
        minimum=0.0,
        help="epsilon for FreeLB adversarial training",
    )
    alpha = FloatField(
        value=0.3,
        minimum=0.0,
        help="alpha for FreeLB adversarial training",
    )
    adv_k = IntField(value=3, minimum=1, help="FreeLB adversarial training times")


@register("adv_method", "free_lb")
class FreeLBAdvMethod(AdvMethod):
    """free_lb adversarial training method"""

    def __init__(self, model: nn.Module, config: FreeLBAdvMethodConfig):
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
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.grad_backup:
                param.grad = (param.grad + self.grad_backup[name]) / 2

    def training_step(self, imodel, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Performs a training step using the FreeLB adversarial method.

        In FreeLB, gradients are accumulated over K steps. Under AMP, gradients are scaled.
        This implementation correctly manually accumulates scaled gradients and allows
        `optimizer.step()` to unscale them.

        Args:
            imodel: The integrated model instance (LightningModule).
            batch: A dictionary containing the mini-batch inputs.
            batch_idx: The index of the current batch.

        Returns:
            Tuple[torch.Tensor, Dict]: A tuple containing the loss and the loss log.
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
        model_grads = {}

        for t in range(self.config.adv_k):
            optimizer.zero_grad()
            result = imodel.model.training_step(batch)
            loss, loss_log = imodel.calc_loss(result, batch, rt_config=rt_config)

            # Compute gradients scaled by AMP (averaged over K steps)
            imodel.manual_backward(loss / self.config.adv_k)

            # Accumulate the pure (but potentially AMP-scaled) gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if name not in model_grads:
                        model_grads[name] = param.grad.clone()
                    else:
                        model_grads[name] += param.grad

            # Calculate next perturbation based on current step's gradient
            if t < self.config.adv_k - 1:
                self.attack(is_first_attack=False)

        # 2. Restore clean embeddings
        self.restore()

        # 3. Inject the accumulated gradients back into the model
        optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            if name in model_grads:
                param.grad = model_grads[name]

        # 4. Step optimizer (Lightning handles AMP unscaling internally via the scaler)
        imodel.clip_gradients(
            optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        optimizer.step()
        imodel.lr_schedulers().step()

        return loss, loss_log
