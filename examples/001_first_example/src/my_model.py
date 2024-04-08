# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

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
from transformers.models.bert.modeling_bert import BertModel

from dlk.nn.base_module import BaseModel
from dlk.utils.register import register, register_module_name


@cregister("model", "my_model")
class BasicModelConfig(Base):
    """
    my simple classfiction model
    """

    bert_model_path = StrField(value=MISSING, help="the bert model path")
    bert_dim = IntField(value=768, help="the bert model dim")
    label_num = IntField(value=2, help="the label num")


@register("model", "my_model")
class BasicModel(BaseModel):
    """Basic encode decode Model"""

    def __init__(self, config: BasicModelConfig, checkpoint):
        super().__init__()
        self.model: BertModel = BertModel.from_pretrained(config.bert_model_path)
        self.classifier = torch.nn.Linear(config.bert_dim, config.label_num)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns: the outputs
        """
        outputs = self.model(
            input_ids=inputs["input_ids"],
            use_cache=None,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=False,
        )

        sequence_output, all_hidden_states, all_self_attentions = (
            outputs[0],
            outputs[2],
            outputs[3],
        )
        inputs["logits"] = self.classifier(sequence_output[:, 0, :])
        return inputs

    def predict_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward(inputs)

    def training_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward(inputs)

    def validation_step(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self.forward(inputs)
