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


@cregister("model", "token_enc_dec")
class TokenEncDecModelConfig(TokenGenBaseConfig):
    src_tokenizer = StrField(
        value=None,
        additions=[None],
        help="the source tokenizer file, if share_embedding is True, src_tokenizer should be None",
    )
    model_type = StrField(
        value="token_enc_dec",
        options=["dec_only", "token_enc_dec", "media_enc_token_dec"],
        help="the model type",
    )
    add_bos_token = BoolField(
        value=False,
        help="whether to add the begin-of-sentence token mannually, the most case is False, because the tokenizer will add it automatically",
    )

    src_embedding_dim = IntField(
        value=None,
        additions=[None],
        help="the target embedding dim, if set to None, will use the `tgt_embedding_dim`",
    )
    decoder_hidden_size = IntField(
        value=None,
        additions=[None],
        help="the hidden size of the decoder, if set to None, will use the `tgt_embedding_dim`",
    )
    max_len_a_ratio = FloatField(
        value=0.0,
        help="the max length ratio(base src), the real max_len = min(max_len_a_ratio * src_len + max_len_b, max_len)",
    )
    max_len_b = IntField(
        value=5,
        help="the targe max length, the real max_len = min(max_len_a_ratio * src_len + max_len_b, max_len)",
    )
    max_len = IntField(
        value=100,
        help="the maximum length, the real max_len = min(max_len_a_ratio * src_len + max_len_b, max_len)",
    )
    match_source_len = BoolField(
        value=False, help="outputs should match the sourcelength (default: False)"
    )


@register("model", "token_enc_dec")
class TokenEncDecModel(TokenGenBase):
    def __init__(self, config: TokenEncDecModelConfig, checkpoint):
        """Generates translations of a given source sentence.
        Args:
        """
        super().__init__(config, checkpoint)
        self.config = config
        if self.config.src_embedding_dim is None:
            self.config.src_embedding_dim = self.config.tgt_embedding_dim

        if config.share_embedding:
            self.src_vocab_size = self.tgt_vocab_size
        else:
            self.src_vocab_size = Tokenizer.from_file(
                config.src_tokenizer
            ).get_vocab_size()

            self.src_embedding = nn.Embedding(
                self.src_vocab_size, self.config.src_embedding_dim
            )

        if self.config.src_embedding_dim is None:
            self.config.src_embedding_dim = self.config.tgt_embedding_dim
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
        if self.config.share_embedding:
            src_embedding = (
                self.embedding.forward(inputs[self.config.input_map.encoder_input_ids])
                * self.embedding_scale
            )
        else:
            src_embedding = (
                self.src_embedding.forward(
                    inputs[self.config.input_map.encoder_input_ids]
                )
                * self.embedding_scale
            )
        inputs[self.encoder.config.input_map.encoder_input_embedding] = src_embedding
        encoder_outs = self.encoder.training_step(inputs)

        if self.config.share_embedding:
            decoder_embedding = (
                self.embedding.forward(
                    encoder_outs[self.config.input_map.decoder_input_ids]
                )
                * self.embedding_scale
            )
        else:
            decoder_embedding = (
                self.tgt_embedding.forward(
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

    @torch.no_grad()
    def generate(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate tokens.

        Args:
            inputs: one mini-batch inputs
        """
        decoder_past_cache = None
        src_tokens = inputs[self.config.input_map.encoder_input_ids]
        inputs[self.config.output_map.decoder_target_ids] = inputs[
            self.config.input_map.decoder_input_ids
        ]
        # length of the source text being the character length except EndOfSentence and pad
        src_lengths = (
            (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        )

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
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

        max_len: int = -1
        if self.config.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.config.max_len_a_ratio * src_len + self.config.max_len_b),
                self.config.max_len - 1,
            )
        assert (
            self.config.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"

        if self.config.share_embedding:
            src_embedding = self.embedding.forward(src_tokens) * self.embedding_scale
        else:
            src_embedding = (
                self.src_embedding.forward(src_tokens) * self.embedding_scale
            )

        inputs[self.encoder.config.input_map.encoder_input_embedding] = src_embedding

        encoder_outs = self.encoder.forward(inputs)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()

        encoder_outs = self.encoder.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        inputs[self.config.output_map.generated] = self._generate_tokens(
            bsz, beam_size, max_len, encoder_outs, src_lengths
        )
        return inputs
