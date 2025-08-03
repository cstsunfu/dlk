# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from . import BaseLoss, BaseLossConfig

logger = logging.getLogger(__name__)


class MultiLabelCategoricalCrossentropy(nn.Module):
    """Implements an enhanced multi-label categorical cross-entropy loss.

    This loss is designed for multi-label classification tasks, especially those
    with soft labels (where target probabilities are between 0 and 1). This
    implementation builds upon the original concept by adding support for label
    smoothing and sample masking.

    The theoretical basis is described in these articles:
        - https://kexue.fm/archives/7359
        - https://kexue.fm/archives/9064
        def multilabel_categorical_crossentropy_pytorch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
            '''
            PyTorch implementation of the multi-label cross-entropy loss from https://kexue.fm/archives/7359.

            说明 (Notes):
                1. y_true and y_pred must have the same shape. y_true contains soft labels (probabilities from 0 to 1).
                2. y_pred should be raw logits from the model (real numbers), without any activation like sigmoid.
                3. For prediction, labels with y_pred > 0 are considered positive.
            '''
            # The math is: log(1 + Σ(1-y_true)*exp(y_pred)) + log(1 + Σ(y_true)*exp(-y_pred))

            # Ensure inputs are tensors
            y_pred = torch.as_tensor(y_pred)
            y_true = torch.as_tensor(y_true)

            # (1) Masking for padded values (optional but good practice)
            # This step is identical to the original if you expect -inf for padding.
            # In many PyTorch use cases, you might handle padding with a separate attention mask.
            # If not handling -inf, this mask will be all True.
            y_mask = (y_pred != -float('inf'))

            # (2) Calculate terms for positive and negative parts
            # To prevent log(0) issues, which result in -inf, we can add a small epsilon.
            # However, torch.log(0) gives -inf, which works correctly with the masking and logsumexp.
            # So, no epsilon is strictly needed if the logic holds.

            # Negative part: y_pred + log(1 - y_true)
            y_neg = y_pred + torch.log(1 - y_true)
            # Positive part: -y_pred + log(y_true)
            y_pos = -y_pred + torch.log(y_true)

            # Apply mask to zero out the contribution of masked elements
            # We set them to -inf so that exp(-inf) = 0 in logsumexp
            y_neg = torch.where(y_mask, y_neg, -float('inf'))
            y_pos = torch.where(y_mask, y_pos, -float('inf'))

            # (3) The log(1 + sum(exp(...))) trick
            # Create a zero tensor of shape (batch_size, 1)
            zeros = torch.zeros_like(y_pred[..., :1])

            # Concatenate the zeros to the last dimension
            y_neg = torch.cat([y_neg, zeros], dim=-1)
            y_pos = torch.cat([y_pos, zeros], dim=-1)

            # (4) Compute logsumexp
            neg_loss = torch.logsumexp(y_neg, dim=-1)
            pos_loss = torch.logsumexp(y_pos, dim=-1)

            # (5) Return the sum of both losses. The result is a tensor of shape (batch_size,)
            return neg_loss + pos_loss
    Attributes:
        epsilon (float): A small value for numerical stability to prevent log(0).
        label_smoothing (float): A factor in [0.0, 1.0) for label smoothing.
        reduction (str): The reduction method to apply to the final loss.

    """

    def __init__(
        self,
        epsilon: float = 1e-7,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        """Initializes the MultiLabelCategoricalCrossentropy loss module.

        Args:
            epsilon (float, optional): A small value for numerical stability.
                Defaults to 1e-7.
            label_smoothing (float, optional): A factor in [0.0, 1.0) for label
                smoothing. If > 0, the formula y_smooth = y * (1-α) + α/C is
                applied to the ground truth labels. Defaults to 0.0.
            reduction (str, optional): Specifies the reduction to apply to the
                output: 'none' | 'mean' | 'sum'.
                - 'none': no reduction will be applied.
                - 'mean': the sum of the output will be divided by the number
                  of unmasked elements.
                - 'sum': the output will be summed.
                Defaults to 'mean'.
        """
        super(MultiLabelCategoricalCrossentropy, self).__init__()

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"reduction '{reduction}' is not valid. Choose from 'mean', 'sum', or 'none'."
            )
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing value must be in [0, 1), but got {label_smoothing}"
            )

        self.epsilon = epsilon
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the loss function.

        Args:
            y_pred (torch.Tensor): The raw, unactivated output of the model
                (logits). Shape: (N, C, ), where N is the batch
                size and C is the number of classes.
            y_true (torch.Tensor): The ground truth labels, with the same shape
                as `y_pred`. Values should be in the range [0, 1].

        Returns:
            torch.Tensor: The computed loss. A scalar tensor if `reduction` is
            'mean' or 'sum', or a tensor of shape (K,) if `reduction` is 'none',
            where K is the number of unmasked samples.
        """
        # Step 1: Validate shapes
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: y_pred shape {y_pred.shape} != y_true shape {y_true.shape}"
            )

        # Step 2: (Optional) Apply label smoothing.
        # Note: This is applied unconditionally to y_true, affecting both
        # hard and soft labels if label_smoothing > 0.
        if self.label_smoothing > 0:
            num_classes = y_true.shape[-1]
            y_true = (
                y_true * (1.0 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )

        # Alternative label smoothing approach:
        #     alpha = self.label_smoothing
        #     y_true = y_true * (1.0 - alpha) + alpha * (1.0 - y_true)

        # Step 3: Clamp y_true to prevent log(0)
        y_true = torch.clamp(y_true, self.epsilon, 1.0 - self.epsilon)

        # Step 4: Create masks for positive and negative classes
        mask_finite = torch.isfinite(y_pred)
        p_mask = (y_true > self.epsilon) & mask_finite
        n_mask = (y_true < 1.0 - self.epsilon) & mask_finite

        # Step 5: Calculate terms for positive and negative loss components
        y_pos_term = torch.where(p_mask, -y_pred + torch.log(y_true), -torch.inf)
        y_neg_term = torch.where(n_mask, y_pred + torch.log(1.0 - y_true), -torch.inf)

        # Step 6: The 'add a zero' trick. Appending a zero to the last dimension
        # ensures that logsumexp result is at least log(1)=0, preventing -inf
        # loss if no positive/negative classes exist for a given sample.
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pos_term = torch.cat([y_pos_term, zeros], dim=-1)
        y_neg_term = torch.cat([y_neg_term, zeros], dim=-1)

        # Step 7: Aggregate using logsumexp to get per-sample loss
        pos_loss = torch.logsumexp(y_pos_term, dim=-1)
        neg_loss = torch.logsumexp(y_neg_term, dim=-1)
        per_sample_loss = pos_loss + neg_loss

        # Step 8: Apply reduction
        if self.reduction == "mean":
            return per_sample_loss.mean()
        elif self.reduction == "sum":
            return per_sample_loss.sum()
        else:  # 'none'
            return per_sample_loss


@cregister("loss", "multilabel_categorical_crossentropy")
class MultiLabelCategoricalCrossEntropyLossConfig(BaseLossConfig):
    """the cross_entropy loss"""

    epsilon = FloatField(
        value=1e-7, help="a small value for numerical stability to prevent log(0)"
    )
    label_smoothing = FloatField(
        value=0.0, minimum=0.0, maximum=1.0, help="the label smoothing"
    )
    reduction = StrField(
        value="mean",
        options=["mean", "sum", "none"],
        help="the reduction method to apply to the output: 'none' | 'mean' | 'sum'. "
        "'none': no reduction will be applied. "
        "'mean': the sum of the output will be divided by the number of unmasked elements. "
        "'sum': the output will be summed.",
    )


@register("loss", "multilabel_categorical_crossentropy")
class MultiLabelCategoricalCrossEntropyLoss(BaseLoss):
    """for multi class classification"""

    def __init__(self, config: MultiLabelCategoricalCrossEntropyLossConfig):
        super(MultiLabelCategoricalCrossEntropyLoss, self).__init__(config)
        self.config: MultiLabelCategoricalCrossEntropyLossConfig

        self.multi_label_loss = MultiLabelCategoricalCrossentropy(
            epsilon=self.config.epsilon,
            label_smoothing=self.config.label_smoothing,
            reduction=self.config.reduction,
        )

    def _calc(self, result, inputs, rt_config, scale):
        """calc the loss the predict is from result, the ground truth is from inputs

        Args:
            result: the model predict dict
            inputs: the all inputs for model
            rt_config: provide the current training status
                >>> {
                >>>     "current_step": self.global_step,
                >>>     "current_epoch": self.current_epoch,
                >>>     "total_steps": self.num_training_steps,
                >>>     "total_epochs": self.num_training_epochs
                >>> }
            scale: the scale rate for the loss
        Returns:
            loss

        """
        pred = result[self.pred_name]
        target = inputs[self.truth_name]
        pred_shape = pred.shape
        if len(pred_shape) == 3:
            # For 3D input [batch_size, num_classes, sequence_length]
            loss = (
                self.multi_label_loss(
                    y_pred=pred.reshape(-1, pred_shape[-1]),
                    y_true=target.reshape(-1, target.shape[-1]),
                )
                * scale
            )
        else:
            assert len(pred_shape) == 4
            loss = (
                self.multi_label_loss(
                    y_pred=pred.reshape(pred_shape[0] * pred_shape[1], -1),
                    y_true=target.reshape(pred_shape[0] * pred_shape[1], -1),
                )
                * scale
            )

        return loss, {self.config.log_map.loss: loss}


# --- Usage Example ---
if __name__ == "__main__":
    batch_size = 4
    num_classes = 5
    torch.manual_seed(42)  # For reproducible results

    y_pred_logits = torch.randn(batch_size, num_classes)
    y_true_hard = torch.tensor(
        [[1, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 1], [1, 1, 0, 0, 0]],
        dtype=torch.float32,
    )

    print("=" * 20 + " 1. Label Smoothing Example " + "=" * 20)
    ls_factor = 0.1
    loss_fn_ls = MultiLabelCategoricalCrossentropy(label_smoothing=ls_factor)

    # Manually calculate smoothed labels for verification
    smoothed_label_manual = y_true_hard * (1 - ls_factor) + ls_factor / num_classes
    print(f"Original hard label (first sample):\n{y_true_hard[0]}")
    print(
        f"Smoothed label (label_smoothing={ls_factor}) (first sample):\n{smoothed_label_manual[0]}"
    )

    # The loss function internally applies this smoothing
    loss_ls = loss_fn_ls(y_pred_logits, y_true_hard)
    print(f"\nLoss with label smoothing (mean): {loss_ls.item():.4f}")

    # For comparison: without label smoothing
    loss_fn_no_ls = MultiLabelCategoricalCrossentropy()
    loss_no_ls = loss_fn_no_ls(y_pred_logits, y_true_hard)
    print(f"Loss without label smoothing (mean): {loss_no_ls.item():.4f}\n")

    loss_original = origin_multilabel_categorical_crossentropy(
        y_true_hard, y_pred_logits
    )
    print(
        f"Loss using original multilabel_categorical_crossentropy: {loss_original.mean().item():.4f}\n"
    )

    print("=" * 20 + " 2. Soft Label Example " + "=" * 20)
    y_true_soft = torch.rand(batch_size, num_classes)
    loss_fn_soft_ls = MultiLabelCategoricalCrossentropy(label_smoothing=0.1)
    loss_soft = loss_fn_soft_ls(y_pred_logits, y_true_soft)
    print(
        "Note: In this implementation, label smoothing is also applied to soft labels."
    )
    print(
        f"Loss with soft labels and label_smoothing={ls_factor} (mean): {loss_soft.item():.4f}"
    )
