# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import pickle as pkl
from typing import Any, Dict, List, Union

import pandas as pd
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

from dlk.data.postprocessor import BasePostProcessor, BasePostProcessorConfig
from dlk.utils.io import open
from dlk.utils.register import register
from dlk.utils.vocab import Vocabulary

logger = logging.getLogger(__name__)


@cregister("postprocessor", "txt_reg")
class TxtRegPostProcessorConfig(BasePostProcessorConfig):
    """postprocess for text regression"""

    class InputMap:
        logits = StrField(value="logits", help="the output logits")
        value = StrField(value="values", help="the target value of the sample")
        index = StrField(value="_index", help="the index of the sample")

    input_map = NestField(
        value=InputMap,
        help="the input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )

    class OriginInputMap:
        uuid = StrField(value="uuid", help="the uuid or the id of the sample")
        sentence = StrField(
            value="sentence", help="the sentence of the sample(for data_type=single)"
        )
        sentence_a = StrField(
            value="sentence_a", help="the sentence_a of the sample(for data_type=pair)"
        )
        sentence_b = StrField(
            value="sentence_b", help="the sentence_b of the sample(for data_type=pair)"
        )

    origin_input_map = NestField(
        value=OriginInputMap,
        help="the origin input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )
    data_type = StrField(
        value="single", options=["single", "pair"], help="the data type, single or pair"
    )
    log_reg = BoolField(value=False, help="whether to return the log reg")


@register("postprocessor", "txt_reg")
class TxtRegPostProcessor(BasePostProcessor):
    """text regression postprocess"""

    def __init__(self, config: TxtRegPostProcessorConfig):
        super(TxtRegPostProcessor, self).__init__(config)
        self.config = config

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
            logits = outputs[self.config.input_map.logits].detach()
            if self.config.log_reg:
                logits = torch.sigmoid(logits)
            assert len(logits.shape) == 2
            # predict_indexes = list(torch.argmax(logits, 1))
            indexes = list(outputs[self.config.input_map.index])

            if self.config.input_map.value in outputs:
                values = outputs[self.config.input_map.value]
            else:
                values = [0.0] * len(indexes)
            for i, (one_logits, index, value) in enumerate(
                zip(logits, indexes, values)
            ):
                one_ins = {}
                one_origin = origin_data.iloc[int(index)]
                if self.config.data_type == "single":
                    sentence = one_origin[self.config.origin_input_map.sentence]
                    one_ins["sentence"] = sentence
                else:
                    sentence_a = one_origin[self.config.origin_input_map.sentence_a]
                    one_ins["sentence_a"] = sentence_a
                    sentence_b = one_origin[self.config.origin_input_map.sentence_b]
                    one_ins["sentence_b"] = sentence_b

                uuid = one_origin[self.config.origin_input_map.uuid]
                one_ins["uuid"] = uuid
                one_ins["values"] = [float(value)]
                one_ins["predict_values"] = [float(one_logits)]
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
            the named scores, acc

        """
        ses = []
        for one_ins in predicts:
            values = one_ins["values"]
            assert (
                len(values) == 1
            ), "We currently is not support multi values in regression postprocess"
            value = values[0]

            one_predicts = one_ins["predict_values"]
            predict_value = one_predicts[0]
            ses.append((predict_value - value) ** 2)
        real_name = self.loss_name_map(stage)
        return {f"{real_name}_mse": sum(ses) / len(ses)}

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
