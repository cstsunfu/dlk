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

@search_method_config_register("length_constrained_beam_search")
class LengthConstrainedBeamSearchConfig(BaseConfig):
    default_config = {
        "_name": "length_constrained_beam_search",
        "config": {
            "min_len_a_ratio": 0.2,
            "min_len_b": 5, 
            "max_len_a_ratio": 2.0, 
            "max_len_b": 10, 
        }
    }
    def __init__(self, config: Dict):
        super(LengthConstrainedBeamSearchConfig, self).__init__(config)
        config = config['config']
        self.min_len_a_ratio = config['min_len_a_ratio']
        self.min_len_b = config['min_len_b']
        self.max_len_a_ratio = config['max_len_a_ratio']
        self.max_len_b = config['max_len_b']
        self.post_check(config, used=[
            "min_len_a_ratio",
            "max_len_b",
            "max_len_a_ratio",
            "max_len_b",
        ])


@search_method_register("length_constrained_beam_search")
class LengthConstrainedBeamSearch(Search):
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

