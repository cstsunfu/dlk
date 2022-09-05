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

# There are some code copied from fairseq
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license

import math
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from dlk.core.models import model_register, model_config_register
from dlk.core.base_module import BaseModel
from .utils import collate_tokens
from .utils import search
from .utils.ngram_repeat_block import NGramRepeatBlock
from dlk.core.layers.embeddings import embedding_config_register, embedding_register
from dlk.core.initmethods import initmethod_config_register, initmethod_register
from dlk.core.layers.encoders import encoder_config_register, encoder_register
from dlk.core.layers.decoders import decoder_config_register, decoder_register
from dlk.utils.config import BaseConfig, ConfigTool
from tokenizers import Tokenizer


class SpecialVocab(object):
    """You should not init this class from scrach"""
    def __init__(self, config: dict):
        self.tokenizer = Tokenizer.from_file(config['config']['tgt_tokenizer'])
        
        self._eos = config['config']['tgt_eos']
        self._pad = config['config']['tgt_pad']
        self._unk = config['config']['tgt_unk']

    def eos(self):
        """
        Returns:
            eos value
        """
        return self.tokenizer.token_to_id(self._eos)

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


@model_config_register('generator')
class GenerateModelConfig(BaseConfig):
    defaut_config = {
            "embedding@encoder": {
                "_base": "static",
                "config": {
                    "embedding_file": "*@*",
                    "embedding_dim": "*@*", # if the embedding_file is a dict, you should provide the dict trace to embedding
                    "embedding_trace": ".", # default the file itself is the embedding
                    "freeze": False, # is freeze
                    "dropout": 0, #dropout rate
                    "output_map": {
                        "embedding": "embedding"
                    },
                    "input_map": {
                        "input_ids": "encoder_input_ids"
                    },
                },
            },
            "embedding@decoder": {
                "_base": "static",
                "config": {
                    "embedding_file": "*@*",
                    "embedding_dim": "*@*", # if the embedding_file is a dict, you should provide the dict trace to embedding
                    "embedding_trace": ".", # default the file itself is the embedding
                    "freeze": False, # is freeze
                    "dropout": 0, #dropout rate
                    "output_map": {
                        "embedding": "decoder_embedding"
                    },
                    "input_map": {
                        "input_ids": "decoder_input_ids"
                    },
                },
            },
            "decoder": {
                "_base": "transformer_decoder",
                "config": {
                    "input_size": "*@*",
                    "output_size": "*@*",
                    "pool": None,
                    "dropout": "*@*", #the decoder output no need dropout
                    "output_map": {}
                    },
                },
            "encoder": {
                "_base": "transformer_encoder",
                "config": {
                    "output_map": {},
                    "hidden_size": "*@*",
                    "input_size": "*@*",
                    "output_size": "*@*",
                    "num_layers": 1,
                    "dropout": "*@*", # dropout between layers
                    },
                },
            "initmethod": {
                "_base": "range_norm"
                },
            "config": {
                "embedding_dim": "*@*",
                "dropout": "*@*",
                "embedding_file": "*@*",
                "embedding_trace": "token_embedding",
                "beam_size": 1,  # beam width (default: 1)
                "max_len_a": 0, 
                "max_len_b": 100,  # generate sequences of maximum length ax + b, where x is the source length
                "max_len": 100, # the maximum length of the generated output(not including end-of-sentence)
                "min_len": 1, # the minimum length of the generated output(not including end-of-sentence)
                "normalize_scores": True, # normalize scores by the length of the output (default: True)
                "len_penalty": 1.0, # length penalty, where <1.0 favors shorter, >1.0 favors longer sentences (default: 1.0)
                "unk_penalty": 0.0, # unknown word penalty, where <0 produces more unks, >0 produces fewer (default: 0.0)
                "temperature": 1.0, # temperature, where values >1.0 produce more uniform samples and values <1.0 produce sharper samples (default: 1.0)
                "match_source_len": False, # outputs should match the sourcelength (default: False)
                "no_repeat_ngram_size": 0, # prevent ngram repeat
                "search_strategy": None,
                "tgt_eos": "[SEP]",
                "tgt_pad": "[PAD]",
                "tgt_unk": "[UNK]",
                "tgt_tokenizer": "*@*",
                },
            "_link": {
                    "config.embedding_dim": ["embedding.config.embedding_dim",
                        "encoder.config.input_size",
                        "encoder.config.output_size",
                        "encoder.config.hidden_size",
                        "decoder.config.output_size",
                        "decoder.config.input_size"
                        ],
                    "config.dropout": ["encoder.config.dropout", "decoder.config.dropout", "embedding.config.dropout"],
                    "config.embedding_file": ['embedding.config.embedding_file'],
                    "config.embedding_trace": ['embedding.config.embedding_trace']
            },
            "_name": "summary_generate"
        }
    def __init__(self, config):
        super(GenerateModelConfig, self).__init__(config)

        self.encoder, self.encoder_config = self.get_encoder(config["encoder"])
        self.decoder, self.decoder_config = self.get_decoder(config["decoder"])
        self.init_method, self.init_method_config = self.get_init_method(config["initmethod"])
        self.share_embedding = config['config']["share_embedding"]
        self.source_embedding, self.source_embedding_config = self.get_embedding(config["embedding@encoder"])
        if not self.share_embedding:
            self.target_embedding, self.target_embedding_config = self.get_embedding(config["embedding@decoder"]) 
        self.beam_size = config['config']["beam_size"]
        self.decoder_hidden_size = config['config']['hidden_size']
        self.max_len_a = config['config']["max_len_a"]
        self.max_len_b = config['config']["max_len_b"]
        self.max_len = config['config']["max_len"]
        self.min_len = config['config']["min_len"]
        self.normalize_scores = config['config']["normalize_scores"]
        self.len_penalty = config['config']["len_penalty"]
        self.unk_penalty = config['config']["unk_penalty"]
        self.temperature = config['config']["temperature"]
        self.match_source_len = config['config']["match_source_len"]
        self.search_strategy = config['config']["search_strategy"] # TODO: add strategy init
        self.no_repeat_ngram_size = config['config']["no_repeat_ngram_size"]
        self.tgt_dict = SpecialVocab(config) # TODO: add target dictionary
        self.vocab_size = len(self.tgt_dict)

    def get_embedding(self, config: Dict):
        """return the Embedding and EmbeddingConfig

        Args:
            config: the embedding config

        Returns: 
            Embedding, EmbeddingConfig

        """
        return ConfigTool.get_leaf_module(embedding_register, embedding_config_register, "embedding", config)

    def get_init_method(self, config: Dict):
        """return the InitMethod and InitMethodConfig

        Args:
            config: the init method config

        Returns: 
            InitMethod, InitMethodConfig

        """
        return ConfigTool.get_leaf_module(initmethod_register, initmethod_config_register, "init method", config)

    def get_encoder(self, config: Dict):
        """return the Encoder and EncoderConfig

        Args:
            config: the encoder config

        Returns: 
            Encoder, EncoderConfig

        """
        return ConfigTool.get_leaf_module(encoder_register, encoder_config_register, "encoder", config)

    def get_decoder(self, config):
        """return the Decoder and DecoderConfig

        Args:
            config: the decoder config

        Returns: 
            Decoder, DecoderConfig

        """
        return ConfigTool.get_leaf_module(decoder_register, decoder_config_register, "decoder", config)


@model_register('generator')
class GeneratorModel(BaseModel):
    def __init__(self, config: GenerateModelConfig, checkpoint):
        """Generates translations of a given source sentence.
        Args:
        """
        super().__init__()
        self.config = config
        self.source_embedding = self.config.source_embedding(self.config.source_embedding_config)
        if not self.config.share_embedding:
            self.target_embedding = self.config.target_embedding(self.config.target_embedding_config)
        else:
            self.target_embedding = self.source_embedding
        self.encoder = self.config.encoder(self.config.encoder_config)
        self.decoder = self.config.decoder(self.config.decoder_config)
        self.lm_head = nn.Linear(self.config.decoder_hidden_size, self.config.vocab_size, bias=False)
        self.decoder.embedding = self.target_embedding
        if not checkpoint:
            init_method = config.init_method(config.init_method_config)
            self.source_embedding.init_weight(init_method)
            if not self.config.share_embedding:
                self.target_embedding.init_weight(init_method)
            self.encoder.init_weight(init_method)
            self.decoder.init_weight(init_method)
            self.lm_head.apply(init_method)
        self.pad = self.config.tgt_dict.pad()
        self.unk = self.config.tgt_dict.unk()
        self.eos = self.config.tgt_dict.eos()
        if self.config.no_repeat_ngram_size > 0:
            self.repeat_ngram_blocker = NGramRepeatBlock(self.config.no_repeat_ngram_size)
        else:
            self.repeat_ngram_blocker = None

        assert self.config.temperature > 0, "--temperature must be greater than 0"

        self.search = (
            search.BeamSearch(self.config.tgt_dict) if self.config.search_strategy is None else self.config.search_strategy
        )
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
            hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
        )

    def provide_keys(self)->List[str]:
        """return all keys of the dict of the model returned

        This method may no use, so we will remove this.

        Returns:
            all keys
        """
        return []

    def check_keys_are_provided(self, provide: List[str]=[])->None:
        """check this all the submodules required key are provided

        Returns:
            None

        Raises:
            PermissionError

        """
        pass

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns:
            the outputs

        """
        return self.generate(inputs)

    def predict_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do predict for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            the predicts outputs

        """
        return self.generate(inputs)

    def training_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do training for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            the training outputs

        """
        embedding_outputs = self.source_embedding.training_step(inputs)
        encode_outputs = self.encoder.training_step(embedding_outputs)
        decoder_embedding_outputs = self.target_embedding.training_step(encode_outputs)
        decode_outputs = self.decoder.training_step(decoder_embedding_outputs)
        decoder_output_embedding = decode_outputs[self.decoder.get_output_name('decoder_output_embedding')]
        logits = self.lm_head(decoder_output_embedding)
        decode_outputs['logits'] = logits
        return decode_outputs

    def validation_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do validation for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            the validation outputs

        """
        return self.generate(inputs)

    def test_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do test for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            the test outputs

        """
        return self.generate(inputs)

    @torch.no_grad()
    def generate(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
        """
        incremental_states = torch.jit.annotate(Dict[str, Dict[str, Optional[torch.Tensor]]], {})

        src_tokens = inputs["encoder_input_ids"]
        inputs['target_ids'] = inputs["decoder_input_ids"]
        # length of the source text being the character length except EndOfSentence and pad
        src_lengths = (
            (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        )

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size: int = self.config.beam_size

        if "constraints" in inputs is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(inputs.get("constraints", None), beam_size)

        max_len: int = -1
        if self.config.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(int(self.config.max_len_a * src_len + self.config.max_len_b), self.config.max_len - 1)
        assert (
            self.config.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"

        embedding_outputs = self.source_embedding.forward(inputs)
        encoder_outs = self.encoder.forward(embedding_outputs)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()

        encoder_outs = self.encoder.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos
        attn: Optional[torch.Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, torch.Tensor]]],
            [torch.jit.annotate(List[Dict[str, torch.Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

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
            .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

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
                self.decoder.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.encoder.reorder_encoder_out(
                    encoder_outs, reorder_state
                )
            encoder_outs[self.target_embedding.get_input_name('decoder_input_ids')] = tokens[:, :step+1]

            decoder_embedding_outputs = self.target_embedding.forward(encoder_outs)
            decoder_outs = self.decoder.forward(
                    decoder_embedding_outputs
            )

            decoder_output_embedding = decoder_outs[self.decoder.get_output_name('decoder_output_embedding')]
            decoder_output_embedding = decoder_output_embedding[:, 0, :]
            lprobs = self.lm_head(decoder_output_embedding)

            avg_attn_scores = None # TODO:get average attention score

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.config.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            prefix_tokens = inputs.get("prefix_tokens", None) # TODO: prefix tokens
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.config.min_len:
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
                self.search.set_src_lengths(src_lengths)

            if self.repeat_ngram_blocker is not None:
                lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.config.vocab_size),
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
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

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
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
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
            self.search.update_constraints(active_hypos)

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
        inputs['generated'] = finalized
        return inputs

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
        ]  # skip the first index, which is EOS

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

        if self.config.match_source_len:
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
