# Copyright the author(s) of DLK.

# There are many code copied from fairseq
# Copyright (c) Facebook, Inc. and its affiliates.


import copy
import math
from typing import Dict, List, Optional

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
from tokenizers import Tokenizer

from dlk.nn.base_module import BaseModel
from dlk.nn.token_gen_base import SpecialVocab, TokenGenBase, TokenGenBaseConfig
from dlk.utils.ngram_repeat_block import NGramRepeatBlock
from dlk.utils.register import register, register_module_name


@cregister("model", "media_enc_token_dec")
class MediaEncTokenDecModelConfig(TokenGenBaseConfig):
    src_tokenizer = StrField(
        value=None,
        additions=[None],
        help="the source tokenizer file, if share_embedding is True, src_tokenizer should be None",
    )
    model_type = StrField(
        value="media_enc_token_dec",
        options=["dec_only", "token_enc_dec", "media_enc_token_dec"],
        help="the model type",
    )
    add_bos_token = BoolField(
        value=False,
        help="whether to add the begin-of-sentence token mannually, the most case is False, because the tokenizer will add it automatically",
    )

    decoder_hidden_size = IntField(
        value=None,
        additions=[None],
        help="the hidden size of the decoder, if set to None, will use the `tgt_embedding_dim`",
    )
    max_len = IntField(
        value=100,
        help="the maximum length",
    )
    min_len = IntField(
        value=1,
        help="the minimum length of the generated output(not including end-of-sentence)",
    )


@register("model", "media_enc_token_dec")
class MediaEncTokenDecModel(TokenGenBase):
    def __init__(self, config: MediaEncTokenDecModelConfig, checkpoint):
        """Generates translations of a given source sentence.
        Args:
        """
        super().__init__(config, checkpoint)
        self.config = config
        if self.config.decoder_hidden_size is None:
            self.config.decoder_hidden_size = self.config.tgt_embedding_dim

    def training_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do training for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            the training outputs

        """
        decoder_input_ids = inputs[self.config.input_map.decoder_input_ids]
        bsz, _ = decoder_input_ids.shape
        if self.config.add_bos_token:
            decoder_input_ids = torch.concat(
                [
                    torch.LongTensor([self.bos] * bsz)
                    .view(-1, 1)
                    .to(decoder_input_ids),
                    decoder_input_ids,
                ],
                -1,
            )
            inputs[self.config.input_map.decoder_input_ids] = decoder_input_ids

        decoder_target_ids = torch.concat(
            [
                decoder_input_ids[:, 1:],
                torch.LongTensor([self.eos] * bsz).view(bsz, -1).to(decoder_input_ids),
            ],
            -1,
        )
        encoder_outs = self.encoder.training_step(inputs)

        decoder_embedding = (
            self.embedding.forward(
                encoder_outs[self.config.input_map.decoder_input_ids]
            )
            * self.embedding_scale
        )
        encoder_outs[
            self.decoder.config.input_map.decoder_input_embedding
        ] = decoder_embedding
        decoder_outs = self.decoder.training_step(encoder_outs)

        decoder_output_embedding = decoder_outs[
            self.decoder.config.output_map.decoder_output_embedding
        ]
        logits = self.lm_head(decoder_output_embedding)
        decoder_outs[self.config.output_map.logits] = logits
        decoder_outs[self.config.output_map.decoder_target_ids] = decoder_target_ids
        return decoder_outs

    def generate(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate translations. Match the api of other fairseq generators.

        Args:
            inputs: one mini-batch inputs
        """
        decoder_input_ids = inputs[self.config.input_map.decoder_input_ids]
        bsz, _ = decoder_input_ids.shape

        inputs[self.config.output_map.decoder_target_ids] = decoder_input_ids
        beam_size: int = self.config.beam_size

        if (
            "constraints" in inputs is not None
            and not self.token_sample.supports_constraints
        ):
            raise NotImplementedError(
                "Target-side constraints were provided, but token_sample method doesn't support them"
            )

        # Initialize constraints, when active
        self.token_sample.init_constraints(inputs.get("constraints", None), beam_size)

        max_len = self.config.max_len - 1
        assert (
            self.config.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"

        encoder_outs = self.encoder.forward(inputs)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(decoder_input_ids.device).long()

        encoder_outs = self.encoder.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        inputs[self.config.output_map.generated] = self._generate_tokens(
            bsz, beam_size, max_len, encoder_outs
        )
        return inputs
