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

# There are many code copy from fairseq.
# Copyright (c) Facebook, Inc. and its affiliates.

from dlk.core.search_methods import Search, search_method_config_register, search_method_register, prefix_constrained_fn_register
import torch
import math
from torch import Tensor
from typing import List, Optional, Dict
from dlk.utils.config import BaseConfig

@search_method_config_register("prefix_constrained_beam_search")
class PrefixConstrainedBeamSearchConfig(BaseConfig):
    default_config = {
        "_name": "prefix_constrained_beam_search",
        "config": {
            "prefix_allowed_tokens_fn": "*@*",
        }
    }
    def __init__(self, config: Dict):
        super(PrefixConstrainedBeamSearchConfig, self).__init__(config)
        config = config['config']
        self.prefix_allowed_tokens_fn = prefix_constrained_fn_register.get(config['prefix_allowed_tokens_fn'])
        self.post_check(config, used=[
            "prefix_allowed_tokens_fn",
        ])


@search_method_register("prefix_constrained_beam_search")
class PrefixConstrainedBeamSearch(Search):
    def __init__(self, tgt_dict, config: PrefixConstrainedBeamSearchConfig):
        super().__init__(tgt_dict)
        self.prefix_allowed_tokens_fn = config.prefix_allowed_tokens_fn
        self.stop_on_max_len = True

    @torch.jit.export
    def apply_mask(self, x, prev_output_tokens, original_batch_idxs):
        beam_size = x.shape[0] // original_batch_idxs.shape[0]
        original_batch_idxs = (
            original_batch_idxs.unsqueeze(-1).repeat((1, beam_size)).flatten().tolist()
        )

        mask = torch.full_like(x, -math.inf)
        for sent_i, (sent, batch_i) in enumerate(
            zip(prev_output_tokens, original_batch_idxs)
        ):
            mask[sent_i, :, self.prefix_allowed_tokens_fn(batch_i, sent)] = 0

        return mask

    @torch.jit.export
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

