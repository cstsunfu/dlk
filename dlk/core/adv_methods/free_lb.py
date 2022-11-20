# Copyright cstsunfu. All rights reserved.
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

import torch.nn as nn
import torch
import numpy as np
import random
import re
from . import adv_method_register, adv_method_config_register, AdvMethod
from typing import Dict, List
from dlk.utils.logger import Logger

logger = Logger.get_logger()

@adv_method_config_register('free_lb')
class FreeLBAdvMethodConfig(object):
    default_config = {
        "_name": "free_lb",
        "config": {
            "embedding_pattern": "model.*embedding.*embedding",
            "epsilon": 1.0,
            "alpha": 0.3,
            "adv_k": 3,
        }
    }
    """Config for FreeLBAdvMethod

    Config Example:
        default_config
    """
    def __init__(self, config: Dict):
        super(FreeLBAdvMethodConfig, self).__init__()
        config = config['config']
        self.embedding_pattern = config['embedding_pattern']
        self.epsilon = config['epsilon']
        self.alpha = config['alpha']
        self.adv_k = config['adv_k']

@adv_method_register('free_lb')
class FreeLBAdvMethod(AdvMethod):
    """Save free_lb decided by config
    """

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
        logger.info(f"There are {len(self.adv_para_name)} paras will be adversarial training.")
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
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.grad_backup:
                param.grad = param.grad + self.grad_backup[name]

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
        rt_config={
            "current_step": imodel.global_step,
            "current_epoch": imodel.current_epoch,
            "total_steps": imodel.num_training_steps,
            "total_epochs": imodel.num_training_epochs
        }
        optimizer.zero_grad()
        seed = random.randint(0, 4e9) # 4e9 < 2e32 - 1
        torch.manual_seed(seed) # NOTE: should fix manual seed for every forward
        np.random.seed(seed)
        result = imodel.model.training_step(batch)
        loss, loss_log = imodel.calc_loss(result, batch, rt_config=rt_config)
        imodel.manual_backward(loss)

        self.backup_grad()
        for t in range(self.config.adv_k):
            self.attack(is_first_attack=(t==0))
            if t == 0:
                optimizer.zero_grad()

            result = imodel.model.training_step(batch)
            loss, loss_log = imodel.calc_loss(result, batch, rt_config=rt_config)
            loss = loss / self.config.adv_k
            imodel.manual_backward(loss)
        self.restore_grad()
        self.restore()
        optimizer.step()

        schedule = imodel.lr_schedulers()
        schedule.step()
        return loss, loss_log
