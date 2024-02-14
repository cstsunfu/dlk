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
    dataclass,
)
from tokenizers import Tokenizer

from dlk.nn.base_module import BaseModel
from dlk.utils.ngram_repeat_block import NGramRepeatBlock
from dlk.utils.register import register, register_module_name


@dataclass
class TokenGenBaseConfig(Base):
    beam_size = IntField(value=1, help="beam search size (default: 1)")
    share_embedding = BoolField(value=True, help="share embedding or not")
    embed_scale = StrField(
        value="none",
        options=["none", "default"],
        help="the embed scale, if `none` will not scale the embedding, if `default` will scale the embedding as sqrt(embed_dim)",
    )
    model_type = StrField(
        value=MISSING,
        options=["dec_only", "token_enc_dec", "media_enc_token_dec"],
        help="the model type",
    )

    tgt_tokenizer = StrField(value=MISSING, help="the target tokenizer file")
    tgt_embedding_dim = IntField(value="???", help="the target embedding dim")
    decoder_hidden_size = IntField(
        value=None,
        additions=[None],
        help="the hidden size of the decoder, if set to None, will use the `tgt_embedding_dim`",
    )
    lm_head_bias = BoolField(value=True, help="the lm head bias")
    normalize_scores = BoolField(
        value=True, help="normalize scores by the length of the output (default: True)"
    )
    len_penalty = FloatField(
        value=1.0,
        help="length penalty, where <1.0 favors shorter, >1.0 favors longer sentences (default: 1.0)",
    )
    unk_penalty = FloatField(
        value=0.0,
        help="unknown word penalty, where <0 produces more unks, >0 produces fewer (default: 0.0)",
    )
    temperature = FloatField(
        value=1.0,
        help="temperature, where values >1.0 produce more uniform samples and values <1.0 produce sharper samples (default: 1.0)",
    )
    no_repeat_ngram_size = IntField(
        value=0,
        minimum=1,
        additions=[0],
        help="prevent ngram repeat, only support 0 or >=2",
    )
    min_len = IntField(
        value=1,
        help="the minimum length of the generated output(not including end-of-sentence)",
    )
    tgt_eos = StrField(value=MISSING, help="the end of sentence token")
    tgt_pad = StrField(value=MISSING, help="the padding token")
    tgt_unk = StrField(value=MISSING, help="the unknown token")
    tgt_bos = StrField(
        value=None,
        additions=[None],
        help="the begin of sentence token, if None, the eos token will be used",
    )

    class InputMap:
        encoder_input_ids = StrField(
            value="encoder_input_ids", help="the encode input ids"
        )
        decoder_input_ids = StrField(
            value="decoder_input_ids", help="the encode input ids"
        )

    input_map = NestField(
        value=InputMap, help="the input map of the token_enc_dec module"
    )

    class OutputMap:
        decoder_target_ids = StrField(value="decoder_target_ids", help="the target ids")
        generated = StrField(value="generated", help="the generated output")
        logits = StrField(value="logits", help="the logits of the token_enc_dec module")

    output_map = NestField(
        value=OutputMap, help="the output map of the token_enc_dec module"
    )

    submodule = SubModule(
        value={},
        suggestions=[
            "encoder",
            "token_gen_decoder",
            "initmethod",
            "token_sample",
        ],
        help="submodules for token_enc_dec model",
    )


class SpecialVocab(object):
    """You should not init this class from scratch"""

    def __init__(self, config):
        self.tokenizer = Tokenizer.from_file(config.tgt_tokenizer)

        self._eos = config.tgt_eos
        self._bos = config.tgt_bos
        self._pad = config.tgt_pad
        self._unk = config.tgt_unk

    def eos(self):
        """
        Returns:
            eos value
        """
        return self.tokenizer.token_to_id(self._eos)

    def bos(self):
        """
        Returns:
            bos value
        """
        if self._bos is None:
            return self.tokenizer.token_to_id(self._eos)
        return self.tokenizer.token_to_id(self._bos)

    def pad(self):
        """
        Returns:
            eos value
        """
        return self.tokenizer.token_to_id(self._pad)

    def unk(self):
        """
        Returns:
            eos value
        """
        return self.tokenizer.token_to_id(self._unk)

    def __len__(self):
        """
        Returns:
            tokenizer vocab size

        """
        return self.tokenizer.get_vocab_size()


class TokenGenBase(nn.Module):
    def __init__(
        self, config: TokenGenBaseConfig, checkpoint, encoder_type: str = "encoder"
    ):
        """Generates translations of a given source sentence.
        Args:
        """
        super().__init__()
        self.config = config
        self.tgt_dict = SpecialVocab(config)
        self.tgt_vocab_size = len(self.tgt_dict)
        self.embedding_scale = (
            math.sqrt(config.tgt_embedding_dim)
            if config.embed_scale == "default"
            else 1.0
        )
        if self.config.decoder_hidden_size is None:
            self.config.decoder_hidden_size = self.config.tgt_embedding_dim

        if config.share_embedding:
            self.embedding = nn.Embedding(
                self.tgt_vocab_size, self.config.tgt_embedding_dim
            )
        else:
            self.tgt_embedding = nn.Embedding(
                self.tgt_vocab_size, self.config.tgt_embedding_dim
            )

        encoder_configs = config._get_modules(encoder_type)
        assert len(encoder_configs) == 1
        if config.share_embedding:
            self.encoder = register.get(
                encoder_type,
                register_module_name(encoder_configs[0]._module_name),
            )(config=encoder_configs[0], embedding=self.embedding)
        else:
            self.encoder = register.get(
                encoder_type,
                register_module_name(encoder_configs[0]._module_name),
            )(config=encoder_configs[0], embedding=self.src_embedding)

        decoder_configs = config._get_modules("token_gen_decoder")
        assert len(decoder_configs) == 1
        if config.share_embedding:
            self.decoder = register.get(
                "token_gen_decoder",
                register_module_name(decoder_configs[0]._module_name),
            )(config=decoder_configs[0], embedding=self.embedding)
        else:
            self.decoder = register.get(
                "token_gen_decoder",
                register_module_name(decoder_configs[0]._module_name),
            )(config=decoder_configs[0], embedding=self.tgt_embedding)

        self.lm_head = nn.Linear(
            self.config.decoder_hidden_size,
            self.tgt_vocab_size,
            bias=self.config.lm_head_bias,
        )
        if not checkpoint:
            init_method_configs = config._get_modules("initmethod")
            if len(init_method_configs) == 0:
                init_method = register.get("initmethod", "default")()
            else:
                assert len(init_method_configs) == 1
                init_method_config = init_method_configs[0]
                init_method = register.get(
                    "initmethod", register_module_name(init_method_config._module_name)
                )(config=init_method_config)
            if self.config.share_embedding:
                self.embedding.apply(init_method)
            else:
                self.src_embedding.apply(init_method)
                self.tgt_embedding.apply(init_method)

            self.encoder.init_weight(init_method)
            self.decoder.init_weight(init_method)
            self.lm_head.apply(init_method)
        self.pad = self.tgt_dict.pad()
        self.unk = self.tgt_dict.unk()
        self.eos = self.tgt_dict.eos()
        self.bos = self.tgt_dict.bos()
        if self.config.no_repeat_ngram_size > 0:
            self.repeat_ngram_blocker = NGramRepeatBlock(
                self.config.no_repeat_ngram_size
            )
        else:
            self.repeat_ngram_blocker = None

        assert self.config.temperature > 0, "--temperature must be greater than 0"

        token_sample_configs = config._get_modules("token_sample")
        if len(token_sample_configs) == 0:
            self.token_sample = register.get("token_sample", "beam_search")(
                self.tgt_dict
            )
        else:
            assert len(token_sample_configs) == 1
            self.token_sample = register.get(
                "token_sample",
                register_module_name(token_sample_configs[0]._module_name),
            )(self.tgt_dict, token_sample_configs[0])

        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        if (
            hasattr(self.token_sample, "needs_src_lengths")
            and self.token_sample.needs_src_lengths
        ):
            assert self.config.model_type == "token_enc_dec", (
                "src_lengths are only needed for token_encode_decode model"
                "but got model_type: {}".format(self.config.model_type)
            )
            self.should_set_src_lengths = True
        else:
            self.should_set_src_lengths = False

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns:
            the outputs

        """
        return self.generate(inputs)

    @torch.no_grad()
    def generate(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate tokens.

        Args:
            inputs: one mini-batch inputs
        """
        raise NotImplementedError

    def predict_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do predict for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            the predicts outputs

        """
        return self.generate(inputs)

    def validation_step(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """do validation for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            the validation outputs

        """
        train_step_outs = self.training_step(copy.deepcopy(inputs))
        result = self.generate(inputs)
        result[self.config.output_map.logits] = train_step_outs[
            self.config.output_map.logits
        ]
        result[self.config.output_map.decoder_target_ids] = train_step_outs[
            self.config.output_map.decoder_target_ids
        ]
        return result

    def test_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do test for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            the test outputs

        """
        train_step_outs = self.training_step(copy.deepcopy(inputs))
        result = self.generate(inputs)
        result[self.config.output_map.logits] = train_step_outs[
            self.config.output_map.logits
        ]
        result[self.config.output_map.decoder_target_ids] = train_step_outs[
            self.config.output_map.decoder_target_ids
        ]
        return result

    def training_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do training for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            the training outputs

        """
        raise NotImplementedError

    def _generate_tokens(self, bsz, beam_size, max_len, encoder_outs, src_lengths=None):
        target_device = encoder_outs[self.config.output_map.decoder_target_ids].device
        decoder_past_cache = None
        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(target_device).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(target_device)
            .long()
            .fill_(self.pad)
        )  # +2 for bos and pad
        tokens[:, 0] = self.bos
        attn: Optional[torch.Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(target_device).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, torch.Tensor]]],
            [torch.jit.annotate(List[Dict[str, torch.Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of information about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
            .unsqueeze(1)
            .type_as(tokens)
            .to(target_device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(target_device)

        reorder_state: Optional[torch.Tensor] = None
        batch_idxs: Optional[torch.Tensor] = None

        original_batch_idxs: Optional[torch.Tensor] = None
        original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                decoder_past_cache = self.decoder.reorder_incremental_state(
                    decoder_past_cache, reorder_state
                )
                encoder_outs = self.encoder.reorder_encoder_out(
                    encoder_outs, reorder_state
                )
            encoder_outs[self.config.input_map.decoder_input_ids] = tokens[
                :, : step + 1
            ]
            if self.config.share_embedding:
                decoder_embedding = (
                    self.embedding.forward(
                        encoder_outs[self.config.input_map.decoder_input_ids][
                            :, step : step + 1
                        ]
                    )
                    * self.embedding_scale
                )
            else:
                decoder_embedding = (
                    self.tgt_embedding.forward(
                        encoder_outs[self.config.input_map.decoder_input_ids][
                            :, step : step + 1
                        ]
                    )
                    * self.embedding_scale
                )
            encoder_outs[
                self.decoder.config.input_map.decoder_input_embedding
            ] = decoder_embedding
            decoder_outs = self.decoder.forward(encoder_outs, decoder_past_cache)
            decoder_past_cache = decoder_outs[
                self.decoder.config.output_map.decoder_past_cache
            ]

            decoder_output_embedding = decoder_outs[
                self.decoder.config.output_map.decoder_output_embedding
            ]
            decoder_output_embedding = decoder_output_embedding[:, -1, :]
            lprobs = self.lm_head(decoder_output_embedding)

            avg_attn_scores = None  # TODO:get average attention score

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            if self.pad != self.eos:
                lprobs[:, self.pad] = -math.inf  # never select pad
            if self.bos != self.eos:
                lprobs[:, self.bos] = -math.inf  # never select bos
            if self.unk != self.eos:
                lprobs[:, self.unk] -= self.config.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            prefix_tokens = encoder_outs.get(
                "prefix_tokens", None
            )  # TODO: prefix tokens
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.config.min_len:
                assert self.config.min_len < max_len
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.token_sample.set_src_lengths(src_lengths)

            if self.repeat_ngram_blocker is not None:
                lprobs = self.repeat_ngram_blocker(
                    tokens[:, : step + 1], lprobs, bsz, beam_size, step
                )

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.token_sample.step(
                step,
                lprobs.view(bsz, -1, self.tgt_vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.token_sample.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                if src_lengths is not None:
                    src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.token_sample.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, torch.Tensor]], finalized[sent]
            )
        return finalized

    def _prefix_tokens(
        self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                :, 0, 1 : step + 1
            ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, torch.Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[torch.Tensor],
        src_lengths,
        max_len: int,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is BOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2]
            if attn is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.config.normalize_scores:
            eos_scores /= (step + 1) ** self.config.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        # [1, 0, 1, 0, 0, 1]
        # [1, 2, 2]
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(bbsz_idx)

        unfin_idx = torch.div(bbsz_idx, beam_size, rounding_mode="trunc")
        sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)

        # Create a set of "{sent}{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # For every finished beam item
        # sentence index in the current (possibly reduced) batch
        seen = (sent << 32) + unfin_idx
        unique_seen: List[int] = torch.unique(seen).tolist()

        # NOTE: only for token encode and token decode model should we consider the match_source_len
        if (
            self.config.model_type == "token_encode_decode"
            and self.config.match_source_len
        ):
            condition = step > torch.index_select(src_lengths, 0, unfin_idx)
            eos_scores = torch.where(condition, torch.tensor(-math.inf), eos_scores)
        sent_list: List[int] = sent.tolist()
        for i in range(bbsz_idx.size()[0]):
            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent_list[i]]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent_list[i]].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": eos_scores[i],
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                    }
                )

        newly_finished: List[int] = []
        for unique_s in unique_seen:
            # check termination conditions for this sentence
            unique_sent: int = unique_s >> 32
            unique_unfin_idx: int = unique_s - (unique_sent << 32)

            if not finished[unique_sent] and self.is_finished(
                step, unique_unfin_idx, max_len, len(finalized[unique_sent]), beam_size
            ):
                finished[unique_sent] = True
                newly_finished.append(unique_unfin_idx)

        return newly_finished

    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False
