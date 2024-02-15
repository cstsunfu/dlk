# Copyright the author(s) of DLK.

# There are many code copy from fairseq.
# Copyright (c) Facebook, Inc. and its affiliates.


from typing import Dict, List, Optional

import torch
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
from torch import Tensor

from dlk.utils.register import register

from . import Sample
from .beam_search import BeamSearch


@cregister("token_sample", "diverse_siblings_search")
class DiverseSiblingsSearchConfig(Base):
    # TODO: diversity_rate need test
    diversity_rate = FloatField(value=0.1, help="the diversity rate")


@register("token_sample", "diverse_siblings_search")
class DiverseSiblingsSearch(Sample):
    """
    Beam search with diverse siblings.

    See "A Simple, Fast Diverse Decoding Algorithm for Neural Generation" for details.
    https://arxiv.org/abs/1611.08562

    1/ Calculate hypotheses for each beam
    2/ Intra-sibling ordering
    3/ Rewrite scores
    4/ Choose top K hypotheses

    if diversity_rate == 0 is equivalent to BeamSearch
    """

    def __init__(self, tgt_dict, config: DiverseSiblingsSearchConfig):
        super().__init__(tgt_dict)
        self.diversity_rate = config.diversity_rate
        self.beam = BeamSearch(tgt_dict)

    def step(
        self,
        step: int,
        lprobs,
        scores,
        prev_output_tokens: Optional[Tensor] = None,
        original_batch_idxs: Optional[Tensor] = None,
    ):
        bsz, beam_size, vocab_size = lprobs.size()
        k = min(
            # Take the best 2 x beam_size predictions. We'll choose the first
            # beam_size of these which don't predict eos to continue with.
            beam_size * 2,
            lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
        )
        s_list: List[Tensor]
        i_list: List[Tensor]
        s_list = [torch.empty(0).to(lprobs) for i in range(beam_size)]
        i_list = [torch.LongTensor().to(device=lprobs.device) for i in range(beam_size)]
        sibling_score = torch.arange(1, k + 1).to(lprobs) * self.diversity_rate

        if step == 0:
            return self.beam.step(step, lprobs, scores)
        lprobs.add_(scores[:, :, step - 1].unsqueeze(-1))

        # 1/ Calculate hypotheses for each beam
        for i in range(beam_size):
            torch.topk(lprobs[:, i, :].view(bsz, -1), k, out=(s_list[i], i_list[i]))
            i_list[i].fmod_(vocab_size)

            # 2/ Intra-sibling ordering by default from topk + 3/ Rewrite scores
            s_list[i].sub_(sibling_score)

        # 4/ Choose top K hypotheses
        indices = torch.stack(i_list, dim=1).view(bsz, -1)

        final_scores = torch.empty(0).to(lprobs)
        final_indices = torch.LongTensor().to(device=lprobs.device)
        final_beams = torch.LongTensor().to(device=lprobs.device)
        (final_scores, final_indices) = torch.topk(
            torch.stack(s_list, dim=1).view(bsz, -1),
            k,
        )

        final_beams = final_indices // k

        for i in range(bsz):
            final_indices[i] = indices[i][final_indices[i]]

        return final_scores, final_indices, final_beams
