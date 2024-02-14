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

from dlk.nn.base_module import BaseModel
from dlk.utils.register import register, register_module_name


@cregister("model", "basic")
class BasicModelConfig(Base):
    """
    The Basic Model, include the embedding, encoder, decoder
    """

    submodule = SubModule(
        value={},
        suggestions=[
            "embedding",
            "encoder",
            "decoder",
            "initmethod",
        ],
        help="submodules for basic model",
    )


@register("model", "basic")
class BasicModel(BaseModel):
    """Basic encode decode Model"""

    def __init__(self, config: BasicModelConfig, checkpoint):
        super().__init__()

        embedding_configs = config._get_modules("embedding")
        if len(embedding_configs) == 0:
            self.embedding = register.get("embedding", "identity")()
        else:
            assert len(embedding_configs) == 1
            self.embedding = register.get(
                "embedding", register_module_name(embedding_configs[0]._module_name)
            )(embedding_configs[0])

        encoder_configs = config._get_modules("encoder")
        if len(encoder_configs) == 0:
            self.encoder = register.get("encoder", "identity")()
        else:
            assert len(encoder_configs) == 1
            self.encoder = register.get(
                "encoder", register_module_name(encoder_configs[0]._module_name)
            )(encoder_configs[0])

        decoder_configs = config._get_modules("decoder")
        if len(decoder_configs) == 0:
            self.decoder = register.get("decoder", "identity")()
        else:
            assert len(decoder_configs) == 1
            self.decoder = register.get(
                "decoder", register_module_name(decoder_configs[0]._module_name)
            )(decoder_configs[0])

        if not checkpoint:
            init_method_configs = config._get_modules("initmethod")
            if len(init_method_configs) == 0:
                init_method = register.get("initmethod", "default")()
            else:
                assert len(init_method_configs) == 1
                init_method_config = init_method_configs[0]
                init_method = register.get(
                    "initmethod", register_module_name(init_method_config._module_name)
                )(init_method_config)
            self.embedding.init_weight(init_method)
            self.encoder.init_weight(init_method)
            self.decoder.init_weight(init_method)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns: the outputs

        """
        embedding_outputs = self.embedding(inputs)
        encode_outputs = self.encoder(embedding_outputs)
        decode_outputs = self.decoder(encode_outputs)
        return decode_outputs

    def predict_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do predict for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: the predicts outputs

        """
        embedding_outputs = self.embedding.predict_step(inputs)
        encode_outputs = self.encoder.predict_step(embedding_outputs)
        decode_outputs = self.decoder.predict_step(encode_outputs)
        return decode_outputs

    def training_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do training for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: the training outputs

        """
        embedding_outputs = self.embedding.training_step(inputs)
        encode_outputs = self.encoder.training_step(embedding_outputs)
        decode_outputs = self.decoder.training_step(encode_outputs)
        return decode_outputs

    def validation_step(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """do validation for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: the validation outputs

        """
        embedding_outputs = self.embedding.validation_step(inputs)
        encode_outputs = self.encoder.validation_step(embedding_outputs)
        decode_outputs = self.decoder.validation_step(encode_outputs)
        return decode_outputs

    def test_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do test for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: the test outputs

        """
        embedding_outputs = self.embedding.test_step(inputs)
        encode_outputs = self.encoder.test_step(embedding_outputs)
        decode_outputs = self.decoder.test_step(encode_outputs)
        return decode_outputs
