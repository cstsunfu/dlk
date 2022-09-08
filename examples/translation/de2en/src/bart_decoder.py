# Copyright 2021 cstsunfu. All rights reserved.
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

import torch
from typing import Dict, List, Set, Callable
from dlk.core.base_module import SimpleModule, BaseModuleConfig
from dlk.core.layers.decoders import decoder_register, decoder_config_register
from dlk.core.modules import module_config_register, module_register
import torch.nn as nn

@decoder_config_register("bart_decoder")
class BartDecoderConfig(BaseModuleConfig):
    default_config = {
            "module": {
                "_base": "bart_decoder",
                },
            "config": {
                "dropout": 0.0,
                "output_map": {},
                "input_map": {}, #  required_key: provide_key
                "pretrained_model_path": "*@*",
                "from_pretrain": True,
                },
            "_link":{
                "config.pretrained_model_path": ["module.config.pretrained_model_path"],
                "config.from_pretrain": ["module.config.from_pretrain"],
                },
            "_name": "bart_decoder",
    }
    """Config for BartDecoder 

    Config Example:
        >>> {
        >>>     "module": {
        >>>         "_base": "bart_decoder",
        >>>     },
        >>>     "config": {
        >>>         "input_size": "*@*",
        >>>         "hidden_size": 0, //default equals to input_size
        >>>         "output_size": "*@*",
        >>>         "dropout": 0.0,
        >>>         "output_map": {},
        >>>         "input_map": {}, // required_key: provide_key
        >>>     },
        >>>     "_link":{
        >>>         "config.input_size": ["module.config.hidden_size"],
        >>>         "config.output_size": ["module.config.output_size"],
        >>>     },
        >>>     "_name": "bart_decoder",
        >>> }
    """
    def __init__(self, config: Dict):
        super(BartDecoderConfig, self).__init__(config)
        self.bart_decoder_config = config["module"]
        config = config['config']
        self.post_check(config, used=[
            "return_logits",
        ])


@decoder_register("bart_decoder")
class BartDecoder(SimpleModule):
    """bart_decoder
    """
    def __init__(self, config: BartDecoderConfig):
        super(BartDecoder, self).__init__(config)
        self._provide_keys = {'logits'}
        self._required_keys = {'embedding'}
        self._provided_keys = set()

        self.config = config

        self.bart_decoder = module_register.get('bart_decoder')(module_config_register.get('bart_decoder')(config.bart_decoder_config))


    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns: 
            None

        """
        self.bart_decoder.init_weight(method)

    @torch.jit.export
    def reorder_incremental_state(
        self,
        encoder_outs: Dict[str, torch.Tensor],
        new_order,
    ):
        decoder_past_cache = encoder_outs.get(self.get_output_name('decoder_past_cache'), None)
        if decoder_past_cache is None:
            return encoder_outs
        encoder_outs[self.get_output_name('decoder_past_cache')] = self.bart_decoder.reorder_incremental_state(decoder_past_cache, new_order)
        return encoder_outs

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        # print("cache: ", inputs.get("encoder_output_embedding", None).shape)
        module_inputs = {
                "decoder_attention_mask": inputs.get(self.get_input_name("decoder_attention_mask"), None),
                "encoder_outputs": inputs.get(self.get_input_name("encoder_output_embedding"), None),
                "decoder_head_mask": inputs.get(self.get_input_name("decoder_head_mask"), None),
                "past_caches": inputs.get(self.get_input_name("decode_past_cache"), None),
                "inputs_embeds": inputs.get(self.get_input_name("decoder_input_embedding"), None),
        }
        hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions = self.bart_decoder(module_inputs)
        inputs[self.get_output_name('decoder_output_embedding')] = hidden_states
        if self._logits_gather.layer_map:
            inputs.update(self._logits_gather([inputs[self.get_output_name('decoder_output_embedding')]]))
        inputs[self.get_output_name('decoder_past_cache')] = next_cache
        return inputs
