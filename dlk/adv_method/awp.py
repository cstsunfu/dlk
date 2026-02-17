# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from intc import FloatField, StrField, cregister

from dlk.utils.register import register

from . import AdvMethod, AdvMethodConfig


@cregister("adv_method", "awp")
class AWPAdvMethodConfig(AdvMethodConfig):
    adv_param_pattern = StrField(
        value="weight",
        help="the pattern of parameter name that awp effect on. default for all weight，or only the weight of higher layer like encoder.layer.11",
    )
    adv_lr = FloatField(value=1.0, help="learning rate or awp")
    adv_eps = FloatField(value=0.2, help="the upper bound of awp perturbation")


@register("adv_method", "awp")
class AWPAdvMethod(AdvMethod):
    """
    Adversarial Weight Perturbation (AWP)
    Paper: https://arxiv.org/abs/2004.05874
    """

    def __init__(self, model: nn.Module, config: AWPAdvMethodConfig):
        super().__init__(model, config)
        self.model = model
        self.config = config
        self.backup = {}
        self.backup_eps = {}

    def attack(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.config.adv_param_pattern in name
            ):
                # 1. 备份干净的权重
                self.backup[name] = param.data.clone()
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 2. 计算动态的上界
                    r_at = self.config.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    with torch.no_grad():
                        param.add_(r_at)
                    # 3. 投影到 epsilon 范围内
                    param.data = torch.min(
                        torch.max(
                            param.data,
                            self.backup[name] - self.config.adv_eps * (norm2 + e),
                        ),
                        self.backup[name] + self.config.adv_eps * (norm2 + e),
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def training_step(self, imodel, batch, batch_idx: int):
        """Performs a training step using Adversarial Weight Perturbation (AWP).

        Args:
            imodel: The integrated model instance (LightningModule).
            batch: A dictionary containing the mini-batch inputs.
            batch_idx: The index of the current batch.

        Returns:
            Tuple[torch.Tensor, Dict]: A tuple containing the adversarial loss and log.
        """
        optimizer = imodel.optimizers()
        rt_config = {
            "current_step": imodel.global_step,
            "current_epoch": imodel.current_epoch,
            "total_steps": imodel.num_training_steps,
            "total_epochs": imodel.num_training_epochs,
        }

        # 1. Compute Loss and gradients for clean data
        optimizer.zero_grad()
        result = imodel.model.training_step(batch)
        loss, _ = imodel.calc_loss(result, batch, rt_config=rt_config)
        imodel.manual_backward(loss)

        # 2. Perturb weights towards gradient ascent direction
        self.attack()

        # 3. Compute Loss and gradients on adversarial weights
        optimizer.zero_grad()
        result = imodel.model.training_step(batch)
        adv_loss, loss_log = imodel.calc_loss(result, batch, rt_config=rt_config)
        imodel.manual_backward(adv_loss)

        # 4. Restore clean weights
        self.restore()

        # 5. Update clean weights using adversarial gradients (AMP handled by Lightning)
        imodel.clip_gradients(
            optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        optimizer.step()
        imodel.lr_schedulers().step()

        return adv_loss, loss_log
