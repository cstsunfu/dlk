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

"""
code is copy from https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/adan.py

Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models[J]. arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677
Implementation adapted from https://github.com/sail-sg/Adan
"""

from typing import Dict, Callable
from torch.optim import Optimizer
import math
import torch
import torch.nn as nn
from . import BaseOptimizer, BaseOptimizerConfig
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules


@config_register("optimizer", 'adan')
@define
class AdanOptimizerConfig(BaseOptimizerConfig):
    name = NameField(value="adan", file=__file__, help="the adan optimizer, ADAptive Nesterov momentum algorithm")
    @define
    class Config(BaseOptimizerConfig.Config):
        lr = FloatField(value=1e-3, checker=float_check(lower=0), help="the learning rate of adan optimizer")
        betas = ListField(value=[0.98, 0.92, 0.99], checker=suggestions(), help="beta 1-2-3 as described in paper - author says most sensitive to beta3 tuning")
        eps = FloatField(value=1e-8, checker=float_check(lower=0), help="the epsilon of adan optimizer")
        weight_decay = FloatField(value=2e-2, checker=float_check(lower=0), help="the weight decay of the optimizer")
    config = NestField(value=Config, converter=nest_converter)

class Adan(Optimizer):
    """
    Implements a pytorch variant of Adan
    Adan was proposed in
    Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models[J]. arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float, flot], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.98, 0.92, 0.99))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay (L2 penalty) (default: 0)
        no_prox (bool): how to perform the decoupled weight decay (default: False)
    """

    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.98, 0.92, 0.99),
            eps=1e-8,
            weight_decay=0.0,
            no_prox=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, no_prox=no_prox)
        super(Adan, self).__init__(params, defaults)

    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    # Exponential moving average of gradient difference
                    state['exp_avg_diff'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        """ Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1.0 - beta1 ** group['step']
            bias_correction2 = 1.0 - beta2 ** group['step']
            bias_correction3 = 1.0 - beta3 ** group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['pre_grad'] = grad.clone()

                exp_avg, exp_avg_sq, exp_avg_diff = state['exp_avg'], state['exp_avg_diff'], state['exp_avg_sq']
                grad_diff = grad - state['pre_grad']

                exp_avg.lerp_(grad, 1. - beta1)  # m_t
                exp_avg_diff.lerp_(grad_diff, 1. - beta2)  # diff_t (v)
                update = grad + beta2 * grad_diff
                exp_avg_sq.mul_(beta3).addcmul_(update, update, value=1. - beta3)  # n_t

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction3)).add_(group['eps'])
                update = (exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2).div_(denom)
                if group['no_prox']:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(update, alpha=-group['lr'])
                else:
                    p.add_(update, alpha=-group['lr'])
                    p.data.div_(1 + group['lr'] * group['weight_decay'])

                state['pre_grad'].copy_(grad)

        return loss

@register("optimizer", "adan")
class AdanOptimizer(BaseOptimizer):
    """wrap for optim.SGD"""
    def __init__(self, model: nn.Module, config: AdanOptimizerConfig):
        super(AdanOptimizer, self).__init__(model, config, Adan)
