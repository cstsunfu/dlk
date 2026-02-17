# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from typing import Any, Dict, List, Optional

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
from torchmetrics.functional.text import bleu_score

from dlk.data.postprocessor import BasePostProcessor, BasePostProcessorConfig
from dlk.utils.register import register, register_module_name
from dlk.utils.tokenizer_util import load_fast_tokenizer

logger = logging.getLogger(__name__)


@cregister("postprocessor", "token_generate")
class TokenGeneratePostProcessorConfig(BasePostProcessorConfig):
    """token generate postprocessor config"""

    class InputMap:
        logits = StrField(value="logits", help="the output logits")
        decoder_target_ids = StrField(value="decoder_target_ids")
        generated = StrField(value="generated")
        index = StrField(value="_index", help="the index of the sample")

    skip_special_tokens = BoolField(
        value=True, help="When decode the tokens, skip the special tokens"
    )
    tokenizer = StrField(value=MISSING, help="the tokenizer config file path")
    return_all_generations = BoolField(
        value=False, help="return all generations, or only the top one"
    )

    class OriginInputMap:
        uuid = StrField(value="uuid", help="the uuid or the id of the sample")
        input = StrField(value="input", help="the prompt for generation")
        target = StrField(value="target", help="the target generation")

    origin_input_map = NestField(
        value=OriginInputMap,
        help="""
        the origin input map of the processor,
        the key is the name of the processor needed key,
        the value is the provided data provided key
        """,
    )

    input_map = NestField(
        value=InputMap,
        help="the input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )


@register("postprocessor", "token_generate")
class TokenGeneratePostProcessor(BasePostProcessor):
    """token generate postprocess"""

    def __init__(self, config: TokenGeneratePostProcessorConfig):
        super(TokenGeneratePostProcessor, self).__init__(config)
        self.config = config
        self.tokenizer = load_fast_tokenizer(self.config.tokenizer)

    def _get_origin_data(self, one_origin: Dict) -> Dict:
        """

        Args:
            one_origin: the original data

        Returns:
            the gather origin data

        """
        origin = {}
        origin["input"] = one_origin[self.config.origin_input_map.input]
        origin["uuid"] = one_origin[self.config.origin_input_map.uuid]
        origin["target"] = one_origin[self.config.origin_input_map.target]
        return origin

    def wrap_predict_one_batch(
        self,
        stage: str,
        batch_output: Dict,
        origin_data: Any,
        rt_config: Dict,
    ) -> List:
        """Process the model predict to human readable format"""
        return self.predict_one_batch(stage, batch_output, origin_data, rt_config)

    def predict_one_batch(
        self, stage, batch_output: Dict, origin_data: Any, rt_config
    ) -> List:
        """Process the model predict to human readable format for one batch"""
        results = []
        indexes = list(batch_output[self.config.input_map.index])

        batch_generated = batch_output[self.config.input_map.generated]
        for i, (index, generated) in enumerate(zip(indexes, batch_generated)):
            one_origin = self._get_origin_row(origin_data, index)
            one_ins = self._get_origin_data(one_origin)

            generate_result = []
            for j, (one_generate) in enumerate(generated):
                generate_sent = self.tokenizer.decode(
                    list(one_generate["tokens"]),
                    skip_special_tokens=self.config.skip_special_tokens,
                )
                generate_result.append(
                    {
                        "generate": generate_sent,
                        "score": float(one_generate["score"]),
                    }
                )
                if self.config.return_all_generations == False:
                    break
            one_ins["generated"] = generate_result
            one_ins["predict_extend_return"] = self.gather_predict_extend_data(
                batch_output, i, self.config.predict_extend_return
            )
            results.append(one_ins)
        return results

    def do_calc_metrics(
        self,
        predicts: List,
        stage: str,
        rt_config: Dict,
    ) -> Dict:
        """calc the scores use the predicts or list_batch_outputs"""
        generates = []
        targets = []
        for one_ins in predicts:
            generates.append(one_ins["generated"][0]["generate"])
            targets.append(one_ins["target"])
        targets_list = [[t] for t in targets]
        bleu_score_value = float(bleu_score(generates, targets_list))

        real_name = self.loss_name_map(stage)
        return {f"{real_name}_bleu": bleu_score_value}
