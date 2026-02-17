# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict

import torch
import torch.nn as nn
from intc import BoolField, FloatField, cregister

from dlk.loss import BaseLoss, BaseLossConfig
from dlk.nn.utils.distributed_utils import gather_tensor_with_grad
from dlk.utils.register import register

logger = logging.getLogger(__name__)


@cregister("loss", "infonce")
class InfoNCELossConfig(BaseLossConfig):
    """Configuration for InfoNCE / NT-Xent Loss."""

    temperature = FloatField(
        value=0.05,
        help="Temperature scaling parameter. Standard values are 0.05 to 0.1.",
    )
    gather_with_grad = BoolField(
        value=True,
        help="Whether to gather embeddings across all GPUs to form a large global batch (Crucial for Scaling).",
    )
    symmetric = BoolField(
        value=False,
        help="Whether to compute symmetric loss (L_a2b + L_b2a) / 2. Useful for CLIP (Image-Text).",
    )


@register("loss", "infonce")
class InfoNCELoss(BaseLoss):
    """InfoNCE (Normalized Temperature-scaled Cross Entropy) Loss."""

    def __init__(self, config: InfoNCELossConfig):
        super(InfoNCELoss, self).__init__(config)
        self.config = config
        self.cross_entropy = nn.CrossEntropyLoss(reduction=self.config.reduction)

    def _calc(self, result: Dict, inputs: Dict, rt_config: Dict, scale: float):
        """Calculates the contrastive loss.

        Args:
            result (Dict): Model prediction containing 'anchor_emb' and 'positive_emb'.
            inputs (Dict): Original inputs.
            rt_config (Dict): Runtime config.
            scale (float): Scale factor for the loss.

        Returns:
            Tuple[torch.Tensor, Dict]: Tuple of (loss tensor, logging dict).
        """
        anchor_emb = result["anchor_emb"]  # [B_local, D]
        positive_emb = result["positive_emb"]  # [B_local, D]

        if self.config.gather_with_grad:
            # Scale up negative pool by gathering across all GPUs
            anchor_emb_all = gather_tensor_with_grad(anchor_emb)  # [B_global, D]
            positive_emb_all = gather_tensor_with_grad(positive_emb)  # [B_global, D]
        else:
            anchor_emb_all = anchor_emb
            positive_emb_all = positive_emb

        # Calculate similarity logits: [B_local, B_global]
        logits = (
            torch.matmul(anchor_emb, positive_emb_all.t()) / self.config.temperature
        )

        # Create diagonal labels
        batch_size = anchor_emb.shape[0]
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        labels = torch.arange(
            rank * batch_size,
            (rank + 1) * batch_size,
            device=anchor_emb.device,
            dtype=torch.long,
        )

        # Forward loss (Anchor -> Positive)
        loss = self.cross_entropy(logits, labels)

        # Symmetric loss (Positive -> Anchor)
        if self.config.symmetric:
            logits_sym = (
                torch.matmul(positive_emb, anchor_emb_all.t()) / self.config.temperature
            )
            loss_sym = self.cross_entropy(logits_sym, labels)
            loss = (loss + loss_sym) / 2.0

        return loss * scale, {self.config.log_map.loss: loss.detach()}
