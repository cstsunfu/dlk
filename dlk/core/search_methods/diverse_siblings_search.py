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

from dlk.core.search_methods import Search, search_method_config_register, search_method_register
from dlk.core.search_methods.beam_search import BeamSearch
import torch
import math
from torch import Tensor
from typing import List, Optional, Dict
from dlk.utils.config import BaseConfig

@search_method_config_register("diverse_siblings_search")
class DiverseSiblingsSearchConfig(BaseConfig):
    default_config = {
        "_name": "diverse_siblings_search",
        "config": {
            "diversity_rate": 0.1, # TODO: Need test
        }
    }
    def __init__(self, config: Dict):
        super(DiverseSiblingsSearchConfig, self).__init__(config)
        config = config['config']
        self.diversity_rate = config['diversity_rate']
        self.post_check(config, used=[
            "diversity_rate",
        ])


@search_method_register("diverse_siblings_search")
class DiverseSiblingsSearch(Search):
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
