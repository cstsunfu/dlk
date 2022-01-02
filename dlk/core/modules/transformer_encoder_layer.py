# Copyright 2021 cstsunfu and the HuggingFace Inc. team.. All rights reserved.
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
import torch.nn as nn
from dlk.core.modules import Module
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from . import module_register, module_config_register, Module, ACT2FN
from dlk.utils.config import BaseConfig

@module_config_register("transformer_encoder_layer")
class TransformerEncoderLayerConfig(BaseConfig):
    """Config for TransformerEncoderLayer

{
    "config": {
        "input_size": 256,
        "num_heads": 2,
        "dropout": 0.0, //the module output no need dropout
        "bias": true, // use bias or not in transformer_encoder_layer , if set to false, all the bias will be set to 0
        "is_decoder": false, // if is decoder will return the past_key_value
    },
    "module":{
        "_base": "transformer_attention",
        "config": {
            "input_size": 256,
            "num_heads": 2,
            "dropout": 0.0, //the module output no need dropout
            "bias": true, // use bias or not in transformer_attention , if set to false, all the bias will be set to 0
            "is_decoder": false, // if is decoder will return the past_key_value
        },
    }
    "_name": "transformer_encoder_layer",
}
    Config Example:
        >>> {
        >>>     "config": {
        >>>         "input_size": 256,
        >>>         "num_heads": 2,
        >>>         "dropout": 0.0, //the module output no need dropout
        >>>         "bias": true, // use bias or not in transformer_encoder_layer , if set to false, all the bias will be set to 0
        >>>         "is_decoder": false, // if is decoder will return the past_key_value
        >>>     },
        >>>     "_name": "transformer_encoder_layer",
        >>> }
    """
    def __init__(self, config: Dict):
        super(TransformerEncoderLayerConfig, self).__init__(config)
        config = config['config']
        self.input_size = config['input_size']
        self.num_heads = config['num_heads']
        self.dropout = float(config['dropout'])
        self.bias = config['bias']
        self.is_decoder = config['is_decoder']
        self.post_check(config, used=[
            "input_size",
            "num_heads",
            "dropout",
            "bias",
            "is_decoder",
        ])


@module_register("transformer_encoder_layer")
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: TransformerEncoderLayerConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
