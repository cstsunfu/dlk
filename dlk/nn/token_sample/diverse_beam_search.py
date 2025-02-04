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


@cregister("token_sample", "diverse_beam_search")
class DiverseBeamSearchConfig(Base):
    num_groups = IntField(value="???", help="the number of diverse groups")
    diversity_strength = FloatField(
        value=0.5,
        help="the strength of diversity, from 0.2 to 0.8 work well for most tasks( https://arxiv.org/pdf/1610.02424.pdf",
    )


@register("token_sample", "diverse_beam_search")
class DiverseBeamSearch(Sample):
    """Diverse Beam Search.

    See "Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence
    Models" for details.

    We only implement the Hamming Diversity penalty here, which performed best
    in the original paper.
    """

    def __init__(self, tgt_dict, config: DiverseBeamSearchConfig):
        super().__init__(tgt_dict)
        self.num_groups = config.num_groups
        self.diversity_strength = -config.diversity_strength
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
        if beam_size % self.num_groups != 0:
            raise ValueError(
                "DiverseBeamSearch requires --beam to be divisible by the number of groups"
            )

        # initialize diversity penalty
        diversity_buf = torch.zeros(lprobs[:, 0, :].size()).to(lprobs)

        scores_G, indices_G, beams_G = [], [], []
        for g in range(self.num_groups):
            lprobs_g = lprobs[:, g :: self.num_groups, :]
            scores_g = scores[:, g :: self.num_groups, :] if step > 0 else None

            # apply diversity penalty
            if g > 0:
                lprobs_g = torch.add(
                    lprobs_g,
                    other=diversity_buf.unsqueeze(1),
                    alpha=self.diversity_strength,
                )
            else:
                lprobs_g = lprobs_g.contiguous()

            scores_buf, indices_buf, beams_buf = self.beam.step(
                step, lprobs_g, scores_g
            )
            beams_buf.mul_(self.num_groups).add_(g)

            scores_G.append(scores_buf.clone())
            indices_G.append(indices_buf.clone())
            beams_G.append(beams_buf.clone())

            # update diversity penalty
            diversity_buf.scatter_add_(
                1, indices_buf, torch.ones(indices_buf.size()).to(diversity_buf)
            )

        # interleave results from different groups
        scores_buf = torch.stack(scores_G, dim=2).view(bsz, -1)
        indices_buf = torch.stack(indices_G, dim=2).view(bsz, -1)
        beams_buf = torch.stack(beams_G, dim=2).view(bsz, -1)
        return scores_buf, indices_buf, beams_buf
