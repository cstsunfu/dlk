# Copyright the author(s) of DLK.
#
# There are many code copy from fairseq.
# Copyright (c) Facebook, Inc. and its affiliates.

""" Wrapper for ngram_repeat_block cuda extension """
import math
import warnings
from typing import List

import torch
from torch import nn


class NGramRepeatBlock(nn.Module):
    """Wrapper class for calling ngram_repeat_block cuda extension"""

    def __init__(self, no_repeat_ngram_size: int, use_extension: bool = False):
        super().__init__()
        self.use_extension = False
        # TODO: use the cuda extensiton
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def reset_parameters(self):
        pass

    def forward(
        self,
        tokens,
        lprobs,
        bsz: int,
        beam_size: int,
        step: int,
    ):
        """
        Args:
            tokens(Tensor): Input tokens(Bsz*beam, seq_len)
            lprobs(Tensor): likelihood probability,
            Expected to be updated in place.(Bsz*beam, vocab_size)
            bsz(int): batch size
            step(int): current step
            beam_size(int): beam size
            no_repeat_ngram_size(int): Ngram size
        """
        msg = f"expected {bsz *beam_size} got"
        assert tokens.size(0) == bsz * beam_size, f"{msg} {tokens.size(0)}"
        assert lprobs.size(0) == bsz * beam_size, f"{msg} {lprobs.size(0)}"
        return self._no_repeat_ngram(
            tokens,
            lprobs,
            bsz,
            beam_size,
            step,
        )

    def _no_repeat_ngram(self, tokens, lprobs, bsz: int, beam_size: int, step: int):
        """For each hypothesis generate a list of previous ngrams and set associated lprobs to -inf"""
        banned_tokens = [
            torch.jit.annotate(List[int], []) for bbsz_idx in range(bsz * beam_size)
        ]
        # print(f"banned, {step}")
        if step + 2 - self.no_repeat_ngram_size >= 0:
            cpu_tokens: List[List[int]] = tokens.cpu().tolist()
            check_start_pos = step + 2 - self.no_repeat_ngram_size
            for bbsz_idx in range(bsz * beam_size):
                ngram_to_check = cpu_tokens[bbsz_idx][
                    -(self.no_repeat_ngram_size - 1) :
                ]
                # print(f"to check: {ngram_to_check}")
                for i in range(check_start_pos):
                    if (
                        ngram_to_check
                        == cpu_tokens[bbsz_idx][i : i + self.no_repeat_ngram_size - 1]
                    ):
                        # print(f"banned, {cpu_tokens[bbsz_idx][i + self.no_repeat_ngram_size - 1]}")
                        banned_tokens[bbsz_idx].append(
                            cpu_tokens[bbsz_idx][i + self.no_repeat_ngram_size - 1]
                        )
        for bbsz_idx in range(bsz * beam_size):
            lprobs[bbsz_idx][
                torch.tensor(banned_tokens[bbsz_idx], dtype=torch.int64)
            ] = torch.tensor(-math.inf).to(lprobs)
        return lprobs
