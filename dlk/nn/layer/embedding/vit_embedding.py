# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Set

import torch
import torch.nn as nn
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

from dlk.nn.base_module import SimpleModule
from dlk.nn.module.module_vit import Vit, VitConfig
from dlk.utils.register import register


@cregister("embedding", "vit")
class VitEmbeddingConfig(VitConfig):
    """
    the bert like embedding module
    """

    embedding_dim = IntField(value=MISSING, minimum=1, help="the embedding dim")

    class InputMap:
        pixel_values = StrField(value="pixel_values", help="the input ids")

    input_map = NestField(
        value=InputMap, help="the input map of the static embedding module"
    )

    class OutputMap:
        embedding = StrField(value="embedding", help="the embedding")

    output_map = NestField(
        value=OutputMap, help="the output map of the static embedding module"
    )


@register("embedding", "vit")
class VitEmbedding(SimpleModule):
    """Wrap the hugingface transformers"""

    def __init__(self, config: VitEmbeddingConfig):
        super(VitEmbedding, self).__init__(config)
        self.config = config
        self.pretrained_transformers = Vit(config)

    def init_weight(self, method):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        self.pretrained_transformers.init_weight(method)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """get the transformers output as embedding

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        pixel_values = inputs[self.config.input_map.pixel_values]
        (
            sequence_output,
            all_hidden_states,
            all_self_attentions,
        ) = self.pretrained_transformers({"pixel_values": pixel_values})
        inputs[self.config.output_map.embedding] = sequence_output
        return inputs
