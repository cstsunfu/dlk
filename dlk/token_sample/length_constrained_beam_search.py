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
from .beam_search import BeamSearch


@cregister("token_sample", "length_constrained_beam_search")
class LengthConstrainedBeamSearchConfig(Base):
    min_len_a_ratio = FloatField(
        value=0.2,
        help="the minimum length ratio, the real min_len = min_len_a_ratio * src_len + min_len_b",
    )
    min_len_b = IntField(
        value=5,
        help="the minimum length, the real min_len = min_len_a_ratio * src_len + min_len_b",
    )
    max_len_a_ratio = FloatField(
        value=2.0,
        help="the maximum length ratio, the real max_len = max_len_a_ratio * src_len + max_len_b",
    )
    max_len_b = IntField(
        value=10,
        help="the maximum length, the real max_len = max_len_a_ratio * src_len + max_len_b",
    )


@register("token_sample", "length_constrained_beam_search")
class LengthConstrainedBeamSearch(Sample):
    def __init__(self, tgt_dict, config: LengthConstrainedBeamSearchConfig):
        super().__init__(tgt_dict)
        self.min_len_a_ratio = config.min_len_a_ratio
        self.min_len_b = config.min_len_b
        self.max_len_a_ratio = config.max_len_a_ratio
        self.max_len_b = config.max_len_b
        self.beam = BeamSearch(tgt_dict)
        self.needs_src_lengths = True

    def step(
        self,
        step: int,
        lprobs,
        scores,
        prev_output_tokens: Optional[Tensor] = None,
        original_batch_idxs: Optional[Tensor] = None,
    ):
        min_lens = self.min_len_a_ratio * self.src_lengths + self.min_len_b
        max_lens = self.max_len_a_ratio * self.src_lengths + self.max_len_b
        lprobs[step < min_lens, :, self.eos] = -math.inf
        lprobs[step >= max_lens, :, self.eos] = 0
        return self.beam.step(step, lprobs, scores)
