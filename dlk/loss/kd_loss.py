# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from intc import BoolField, DictField, FloatField, cregister

from dlk.loss import BaseLoss, BaseLossConfig
from dlk.utils.register import register


@cregister("loss", "kd")
class KDLossConfig(BaseLossConfig):
    """Configuration for Knowledge Distillation Loss."""

    student_teacher_pair = DictField(
        value={"logits": "teacher_logits"},
        help="Dictionary mapping student prediction key to teacher label key in inputs.",
    )
    temperature = FloatField(
        value=2.0, minimum=0.1, help="Temperature for softening probabilities."
    )
    alpha = FloatField(
        value=0.5,
        minimum=0.0,
        maximum=1.0,
        help="Weight for KD loss. Total Loss = alpha * KD + (1 - alpha) * CE.",
    )
    use_hard_label = BoolField(
        value=True,
        help="Whether to calculate cross entropy with ground truth hard labels.",
    )


@register("loss", "kd")
class KDLoss(BaseLoss):
    """Computes Knowledge Distillation Loss without hardcoding field names."""

    def __init__(self, config: KDLossConfig):
        super().__init__(config)
        self.config = config
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")

        # Parse student-teacher mapping
        assert (
            len(self.config.student_teacher_pair) == 1
        ), "Only one student_teacher_pair is supported."
        self.st_pred_name = list(self.config.student_teacher_pair.keys())[0]
        self.st_teacher_name = self.config.student_teacher_pair[self.st_pred_name]

    def _calc(
        self,
        result: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor],
        rt_config: Dict,
        scale: float,
    ):
        """Calculates KD loss.

        Args:
            result: Model outputs dict.
            inputs: Ground truth dict containing teacher labels and hard labels.
            rt_config: Runtime configuration.
            scale: Scale rate for the loss.

        Returns:
            Tuple of (Total Loss, Loss Dict for logging).
        """
        student_logits = result[self.st_pred_name]
        teacher_logits = inputs[self.st_teacher_name]

        T = self.config.temperature
        alpha = self.config.alpha

        # 1. KD Loss (Soft Labels)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        if self.config.use_hard_label:
            mask = inputs[self.truth_name] != self.config.ignore_index
            student_log_probs = student_log_probs[mask]
            teacher_probs = teacher_probs[mask]

        if student_log_probs.dim() > 2:
            student_log_probs = student_log_probs.view(-1, student_log_probs.size(-1))
            teacher_probs = teacher_probs.view(-1, teacher_probs.size(-1))
        loss_kd = self.kl_loss(student_log_probs, teacher_probs) * (T * T)

        total_loss = alpha * loss_kd
        loss_dict = {f"{self.config.log_map.loss}_kd": loss_kd * scale}

        # 2. Hard Label CE Loss (using BaseLossConfig's pred_truth_pair)
        if self.config.use_hard_label:
            hard_pred = result[self.pred_name]
            hard_truth = inputs[self.truth_name]

            loss_ce = self.ce_loss(hard_pred, hard_truth)
            total_loss += (1.0 - alpha) * loss_ce
            loss_dict[f"{self.config.log_map.loss}_ce"] = loss_ce * scale

        loss_dict[self.config.log_map.loss] = total_loss.detach
        return total_loss * scale, loss_dict
