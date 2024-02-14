# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Set

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

from dlk.nn.base_module import BaseModule
from dlk.utils.register import register


@cregister("decoder", "linear_crf")
class DecoderLinearCRFConfig(Base):
    """the linear_crf decoder module"""

    input_size = IntField(value=MISSING, minimum=1, help="the input size")
    output_size = IntField(value=MISSING, minimum=1, help="the output size")
    reduction = StrField(
        value="mean",
        options=["none", "sum", "mean", "token_mean"],
        help="the reduction method",
    )

    class InputMap:
        embedding = StrField(value="embedding", help="the embedding")
        label_ids = StrField(value="label_ids", help="the label ids")
        attention_mask = StrField(value="attention_mask", help="the attention mask")

    input_map = NestField(value=InputMap, help="the input map of the linear_crf module")

    class OutputMap:
        predict_seq_label = StrField(
            value="predict_seq_label", help="the predict seq label"
        )
        loss = StrField(value="loss", help="the loss")

    output_map = NestField(
        value=OutputMap, help="the output map of the linear_crf module"
    )

    submodule = SubModule(
        {
            "module@linear": {
                "input_size": "@$$.input_size @lambda x: x",
                "output_size": "@$$.output_size @lambda x: x",
            },
            "module@crf": {
                "reduction": "@$$.reduction @lambda x: x",
                "output_size": "@$$.output_size @lambda x: x",
            },
        }
    )


@register("decoder", "linear_crf")
class DecoderLinearCRF(BaseModule):
    """use torch.nn.Linear get the emission probability and fit to CRF"""

    def __init__(self, config: DecoderLinearCRFConfig):
        super(DecoderLinearCRF, self).__init__(config)

        self.linear = register.get("module", "linear")(config["@module@linear"])
        self.crf = register.get("module", "crf")(config["@module@crf"])
        self.config = config

    def init_weight(self, method: Callable):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        self.linear.init_weight(method)
        self.crf.init_weight(method)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do predict, only get the predict labels

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        return self.predict_step(inputs)

    def predict_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do predict, only get the predict labels

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        logits = self.linear(inputs[self.config.input_map.embedding])
        inputs[self.config.output_map.predict_seq_label] = self.crf(
            logits, inputs[self.config.input_map.attention_mask]
        )
        return inputs

    def training_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do training step, get the crf loss

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        logits = self.linear(inputs[self.config.input_map.embedding])
        loss = self.crf.training_step(
            logits,
            inputs[self.config.input_map.label_ids],
            inputs[self.config.input_map.attention_mask],
        )
        inputs[self.config.output_map.loss] = loss
        return inputs

    def validation_step(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """do validation step, get the crf loss and the predict labels

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        logits = self.linear(inputs[self.config.input_map.embedding])
        loss = self.crf.training_step(
            logits,
            inputs[self.config.input_map.label_ids],
            inputs[self.config.input_map.attention_mask],
        )
        inputs[self.config.output_map.loss] = loss
        inputs[self.config.output_map.predict_seq_label] = self.crf(
            logits, inputs[self.config.input_map.attention_mask]
        )
        return inputs
