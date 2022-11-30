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

@search_method_config_register("diverse_beam_search")
class DiverseBeamSearchConfig(BaseConfig):
    default_config = {
        "_name": "diverse_beam_search",
        "config": {
            "num_groups": "*@*",
            "diversity_strength": 0.5, # from 0.2 to 0.8 work well for most tasks( https://arxiv.org/pdf/1610.02424.pdf)
        }
    }
    def __init__(self, config: Dict):
        super(DiverseBeamSearchConfig, self).__init__(config)
        config = config['config']
        self.num_groups = config['num_groups']
        self.diversity_strength = config['diversity_strength']
        self.post_check(config, used=[
            "num_groups",
            "diversity_strength",
        ])


@search_method_register("diverse_beam_search")
class DiverseBeamSearch(Search):
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

    @torch.jit.export
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

