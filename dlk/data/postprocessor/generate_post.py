# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
import os
from typing import Dict, List, Optional

import pandas as pd
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
from torchmetrics.functional.text import bleu_score

from dlk.data.postprocessor import BasePostProcessor, BasePostProcessorConfig
from dlk.utils.register import register, register_module_name

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

        self.tokenizer = Tokenizer.from_file(self.config.tokenizer)

    def _get_origin_data(self, one_origin: pd.Series) -> Dict:
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

    def do_predict(
        self,
        stage: str,
        list_batch_outputs: List[Dict],
        origin_data: pd.DataFrame,
        rt_config: Dict,
    ) -> List:
        """Process the model predict to human readable format

        Args:
            stage: train/test/etc.
            list_batch_outputs: a list of outputs
            origin_data: the origin pd.DataFrame data, there are some data not be able to convert to tensor
            rt_config:
                >>> current status
                >>> {
                >>>     "current_step": self.global_step,
                >>>     "current_epoch": self.current_epoch,
                >>>     "total_steps": self.num_training_steps,
                >>>     "total_epochs": self.num_training_epochs
                >>> }

        Returns:
            all predicts
        """
        results = []
        for outputs in list_batch_outputs:
            indexes = list(outputs[self.config.input_map.index])

            batch_generated = outputs[self.config.input_map.generated]
            for i, (index, generated) in enumerate(zip(indexes, batch_generated)):
                one_origin = origin_data.iloc[int(index)]
                one_ins = self._get_origin_data(one_origin)

                generate_result = []
                for i, (one_generate) in enumerate(generated):
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
                    outputs, i, self.config.predict_extend_return
                )
                results.append(one_ins)
        return results

    def do_calc_metrics(
        self,
        predicts: List,
        stage: str,
        list_batch_outputs: List[Dict],
        origin_data: pd.DataFrame,
        rt_config: Dict,
    ) -> Dict:
        """calc the scores use the predicts or list_batch_outputs

        Args:
            predicts: list of predicts
            stage: train/test/etc.
            list_batch_outputs: a list of outputs
            origin_data: the origin pd.DataFrame data, there are some data not be able to convert to tensor
            rt_config:
                >>> current status
                >>> {
                >>>     "current_step": self.global_step,
                >>>     "current_epoch": self.current_epoch,
                >>>     "total_steps": self.num_training_steps,
                >>>     "total_epochs": self.num_training_epochs
                >>> }

        Returns:
            the named scores

        """
        generates = []
        targets = []
        for one_ins in predicts:
            generates.append(one_ins["generated"][0]["generate"])
            targets.append(one_ins["target"])
        bleu_score_value = float(bleu_score(generates, targets))

        real_name = self.loss_name_map(stage)
        return {f"{real_name}_bleu": bleu_score_value}

    def do_save(
        self,
        predicts: List,
        stage: str,
        list_batch_outputs: List[Dict],
        origin_data: pd.DataFrame,
        rt_config: Dict,
        save_condition: bool = False,
    ):
        """save the predict when save_condition==True

        Args:
            predicts: list of predicts
            stage: train/test/etc.
            list_batch_outputs: a list of outputs
            origin_data: the origin pd.DataFrame data, there are some data not be able to convert to tensor
            rt_config:
                >>> current status
                >>> {
                >>>     "current_step": self.global_step,
                >>>     "current_epoch": self.current_epoch,
                >>>     "total_steps": self.num_training_steps,
                >>>     "total_epochs": self.num_training_epochs
                >>> }
            save_condition: True for save, False for depend on rt_config

        Returns:
            None
        """
        if self.config.start_save_epoch == -1 or self.config.start_save_step == -1:
            self.config.start_save_step = rt_config.get("total_steps", 0) - 1
            self.config.start_save_epoch = rt_config.get("total_epochs", 0) - 1
        if not save_condition and (
            rt_config["current_step"] >= self.config.start_save_step
            or rt_config["current_epoch"] >= self.config.start_save_epoch
        ):
            save_condition = True
        if save_condition:
            save_path = os.path.join(
                self.config.save_root_path, self.config.save_dir.get(stage, "")
            )
            if "current_step" in rt_config:
                save_file = os.path.join(
                    save_path, f"step_{str(rt_config['current_step'])}_predict.json"
                )
            else:
                save_file = os.path.join(save_path, "predict.json")
            logger.info(f"Save the {stage} predict data at {save_file}")
            with open(save_file, "w") as f:
                json.dump(predicts, f, indent=4, ensure_ascii=False)
