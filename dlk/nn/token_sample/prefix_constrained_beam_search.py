# Copyright the author(s) of DLK.

# There are many code copy from fairseq.
# Copyright (c) Facebook, Inc. and its affiliates.


import math
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


@cregister("token_sample", "prefix_constrained_beam_search")
class PrefixConstrainedBeamSearchConfig(Base):
    gen_prefixs_fn = StrField(
        value="???",
        help="the prefix allowed tokens function, registerd in gen_prefixs_fn",
    )


@register("token_sample", "prefix_constrained_beam_search")
class PrefixConstrainedBeamSearch(Sample):
    def __init__(self, tgt_dict, config: PrefixConstrainedBeamSearchConfig):
        super().__init__(tgt_dict)
        self.gen_prefixs_fn = register.get("gen_prefixs_fn", config.gen_prefixs_fn)
        self.stop_on_max_len = True

    def apply_mask(self, x, prev_output_tokens, original_batch_idxs):
        beam_size = x.shape[0] // original_batch_idxs.shape[0]
        original_batch_idxs = (
            original_batch_idxs.unsqueeze(-1).repeat((1, beam_size)).flatten().tolist()
        )

        mask = torch.full_like(x, -math.inf)
        for sent_i, (sent, batch_i) in enumerate(
            zip(prev_output_tokens, original_batch_idxs)
        ):
            mask[sent_i, :, self.gen_prefixs_fn(batch_i, sent)] = 0

        return mask

    def step(
        self,
        step: int,
        lprobs: Tensor,
        scores: Tensor,
        prev_output_tokens: Tensor,
        original_batch_idxs: Tensor,
    ):
        bsz, beam_size, vocab_size = lprobs.size()

        lprobs += self.apply_mask(
            lprobs.view(bsz * beam_size, 1, vocab_size),
            prev_output_tokens,
            original_batch_idxs,
        ).view(bsz, beam_size, vocab_size)

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        top_prediction = torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                # Take the best beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        beams_buf = indices_buf // vocab_size
        indices_buf = indices_buf.fmod(vocab_size)
        return scores_buf, indices_buf, beams_buf
