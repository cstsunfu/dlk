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

import torch.nn as nn
import torch
from typing import Dict, List, Set, Callable
from dlk.core.base_module import SimpleModule, BaseModuleConfig
from dlk.core.layers.encoders import encoder_register, encoder_config_register
from dlk.core.modules import module_config_register, module_register
from dlk.utils.logger import Logger
logger = Logger.get_logger()

@encoder_config_register("bart_encoder")
class BartEncoderConfig(BaseModuleConfig):
    default_config = {
            "module": {
                "_base": "bart_encoder",
            },
            "config": {
                "input_map": {},
                "output_map": {},
                "dropout": 0.0, # dropout between layers
                "pretrained_model_path": "*@*",
                "from_pretrain": True,
                },
            "_link": {
                "config.pretrained_model_path": ["module.config.pretrained_model_path"],
                "config.from_pretrain": ["module.config.from_pretrain"],
                },
            "_name": "bart_encoder",
    }
    """Config for BartEncoder

    Config Example:
            {
                "module": {
                    "_base": "bart_encoder",
                    },
                "config": {
                    "input_map": {},
                    "output_map": {},
                    "input_size": "*@*",
                    "output_size": "*@*",
                    "num_layers": 1,
                    "dropout": "*@*", # dropout between layers
                    },
                "_link": {
                    "config.input_size": ["module.config.input_size"],
                    "config.output_size": ["module.config.output_size"],
                    "config.dropout": ["module.config.dropout"],
                    },
                "_name": "bart_encoder",
            }
    """

    def __init__(self, config: Dict):
        super(BartEncoderConfig, self).__init__(config)
        self.bart_encoder_config = config["module"]
        assert self.bart_encoder_config['_name'] == "bart_encoder"
        self.post_check(config['config'], used=[
            "input_size",
            "output_size",
            "num_layers",
            "return_logits",
            "dropout",
        ])


@encoder_register("bart_encoder")
class BartEncoder(SimpleModule):
    """Wrap for torch.nn.BartEncoder
    """
    def __init__(self, config: BartEncoderConfig):
        super(BartEncoder, self).__init__(config)
        self._provide_keys = {'embedding'}
        self._required_keys = {'embedding', 'attention_mask'}
        self._provided_keys = set()
        self.config = config
        self.bart_encoder = module_register.get('bart_encoder')(module_config_register.get('bart_encoder')(config.bart_encoder_config))

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns: 
            None

        """
        self.bart_encoder.init_weight(method)

    @torch.jit.export
    def reorder_encoder_out(self, encoder_outs: Dict[str, torch.Tensor], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        encoder_output_embedding = encoder_outs.get(self.get_output_name('encoder_output_embedding'), None)
        if encoder_output_embedding is not None:
           encoder_outs[self.get_output_name('encoder_output_embedding')]  = encoder_output_embedding.index_select(1, new_order)

        encoder_head_mask = encoder_outs.get(self.get_output_name('encoder_head_mask'), None)
        if encoder_head_mask is not None:
           encoder_outs[self.get_output_name('encoder_head_mask')]  = encoder_head_mask.index_select(1, new_order)

        encoder_input_embedding = encoder_outs.get(self.get_output_name('encoder_input_embedding'), None)
        if encoder_input_embedding is not None:
           encoder_outs[self.get_output_name('encoder_input_embedding')]  = encoder_input_embedding.index_select(1, new_order)

        return encoder_outs


    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """All step do this

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        module_inputs = {
                "attention_mask": inputs.get(self.get_input_name("encoder_attention_mask"), None),
                "head_mask": inputs.get(self.get_input_name("encoder_head_mask"), None),
                "inputs_embeds": inputs.get(self.get_input_name("encoder_input_embedding"), None),
        }

        sequence_output, all_hidden_states, all_self_attentions = self.bart_encoder(module_inputs)
        inputs[self.get_output_name('encoder_output_embedding')] = sequence_output
        if self._logits_gather.layer_map:
            inputs.update(self._logits_gather(all_hidden_states))
        return inputs
