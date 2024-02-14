# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List

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

from . import Module


@cregister("module", "crf")
class CRFConfig:
    """the ConditionalRandomField config"""

    output_size = IntField(value=MISSING, minimum=0, help="the output size")
    batch_first = BoolField(value=True, help="whether the input is batch first")
    reduction = StrField(
        value="mean",
        options=["none", "sum", "mean", "token_mean"],
        help="the reduction method",
    )


@register("module", "crf")
class ConditionalRandomField(Module):
    """CRF, training_step for training, forward for decode。"""

    def __init__(self, config: CRFConfig):
        super(ConditionalRandomField, self).__init__()
        self.config = config

        self.num_tags = self.config.output_size

        self.transitions = nn.parameter.Parameter(
            torch.randn(self.num_tags, self.num_tags)
        )
        self.start_transitions = nn.parameter.Parameter(torch.randn(self.num_tags))
        self.end_transitions = nn.parameter.Parameter(torch.randn(self.num_tags))

    def init_weight(self, method: Callable):
        """init the weight of transitions, start_transitions and end_transitions

        Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.

        Args:
            method: init method, no use

        Returns:
            None

        """

        nn.init.normal_(self.transitions, -1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def _normalizer_likelihood(self, logits: torch.FloatTensor, mask: torch.ByteTensor):
        """Computes the (batch_size,) denominator term for the log-likelihood.

        The sum of the likelihoods across all possible state sequences.

        Args:
            logits: max_len*batch_size*num_tags
            mask: max_len*batch_size

        Returns:
            batch_size*every sum

        """
        seq_len, batch_size, n_tags = logits.size()
        alpha = logits[0]
        alpha = alpha + self.start_transitions.view(1, -1)

        flip_mask = mask.eq(False)

        for i in range(1, seq_len):
            emit_score = logits[i].view(batch_size, 1, n_tags)
            trans_score = self.transitions.view(1, n_tags, n_tags)
            tmp = alpha.view(batch_size, n_tags, 1) + emit_score + trans_score
            alpha = torch.logsumexp(tmp, 1).masked_fill(
                flip_mask[i].view(batch_size, 1), 0
            ) + alpha.masked_fill(mask[i].eq(True).view(batch_size, 1), 0)

        alpha = alpha + self.end_transitions.view(1, -1)

        return torch.logsumexp(alpha, 1)

    def _gold_score(
        self, logits: torch.FloatTensor, tags: torch.LongTensor, mask: torch.ByteTensor
    ):
        """Compute the score for the gold path.

        Args:
            logits: max_len*batch_size*num_tags
            tags: max_len*batch_size
            mask: max_len*batch_size

        Returns:
            batch_size*every_gold_score

        """
        seq_len, batch_size, _ = logits.size()
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=logits.device)

        # trans_socre [L-1, B]
        mask = mask.eq(True)
        flip_mask = mask.eq(False)
        trans_score = self.transitions[tags[: seq_len - 1], tags[1:]].masked_fill(
            flip_mask[1:, :], 0
        )
        # emit_score [L, B]
        emit_score = logits[
            seq_idx.view(-1, 1), batch_idx.view(1, -1), tags
        ].masked_fill(flip_mask, 0)
        # score [L-1, B]
        score = trans_score + emit_score[: seq_len - 1, :]
        score = score.sum(0) + emit_score[-1].masked_fill(flip_mask[-1], 0)
        st_scores = self.start_transitions.view(1, -1).repeat(batch_size, 1)[
            batch_idx, tags[0]
        ]
        last_idx = mask.long().sum(0) - 1
        ed_scores = self.end_transitions.view(1, -1).repeat(batch_size, 1)[
            batch_idx, tags[last_idx, batch_idx]
        ]
        score = score + st_scores + ed_scores
        # return [B,]
        return score

    def training_step(
        self, logits: torch.FloatTensor, tags: torch.LongTensor, mask: torch.LongTensor
    ):
        """training step, calc the loss

        Args:
            logits: emissions, batch_size*max_len*num_tags
            tags: batch_size*max_len
            mask: batch_size*max_len, mask==0 means padding

        Returns:
            loss

        """
        logits = logits.transpose(0, 1)
        tags = tags.transpose(0, 1).long()
        mask = mask.transpose(0, 1).byte()
        all_path_score = self._normalizer_likelihood(logits, mask)
        gold_path_score = self._gold_score(logits, tags, mask)
        loss = all_path_score - gold_path_score
        return loss.mean()

    def forward(self, logits: torch.FloatTensor, mask: torch.LongTensor):
        """predict step, get the best path

        Args:
            logits: emissions, batch_size*max_len*num_tags
            mask: batch_size*max_len, mask==0 means padding

        Returns:
            batch*max_len

        """
        logits = logits.transpose(0, 1)  # L, B, H
        mask = mask.transpose(0, 1)
        return self._viterbi_decode(logits, mask)

    def _viterbi_decode(
        self, emissions: torch.FloatTensor, mask: torch.LongTensor
    ) -> torch.Tensor:
        """predict step, get the best path

        Args:
            logits: emissions, max_len*batch_size*num_tags
            mask: max_len*batch_size, mask==0 means padding

        Returns:
            batch*max_len

        """
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = torch.jit.annotate(List[int], [])

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask.bool()[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []
        best_tags_list = torch.jit.annotate(List[List[int]], [])

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        output = torch.jit.annotate(List[List[int]], [])
        for tag_list in best_tags_list:
            if len(tag_list) < seq_length:
                tag_list = tag_list + [-1] * (seq_length - len(tag_list))
            output.append(tag_list)
        return torch.tensor(output, dtype=torch.long, device=mask.device)
