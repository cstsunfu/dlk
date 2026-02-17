# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Union

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
from dlk.utils.register import register

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

    def wrap_predict_one_batch(
        self,
        stage: str,
        batch_output: Dict,
        origin_data: Any,
        rt_config: Dict,
    ) -> List:
        """Process the model predict to human readable format

        Args:
            stage: train/test/etc.
            batch_output: model outputs
            origin_data: the origin Any data, there are some data not be able to convert to tensor
            rt_config:
                >>> current status
                >>> {
                >>>     "current_step": self.global_step,
                >>>     "current_epoch": self.current_epoch,
                >>>     "total_steps": self.num_training_steps,
                >>>     "total_epochs": self.num_training_epochs
                >>> }

        Returns:
            the predicts

        """
        batch_output[self.config.input_map.logits] = (
            batch_output[self.config.input_map.logits].detach().cpu().numpy()
        )
        return self.predict_one_batch(stage, batch_output, origin_data, rt_config)

    def predict_one_batch(
        self, stage, batch_output: Dict, origin_data: Any, rt_config
    ) -> List:
        """Process the model predict to human readable format for one batch
        Args:
            stage: train/test/etc.
            batch_output: a dict of outputs
            origin_data: the origin Any data, there are some data not be able to convert to tensor
        Returns:
            the predicts of one batch
        """
        logits = batch_output[self.config.input_map.logits]
        if self.config.log_reg:
            logits = 1 / (1 + np.exp(-logits))

        assert (
            len(logits.shape) == 2
        ), f"Logits should be 2D array, got shape {logits.shape}"
        # predict_indexes = list(torch.argmax(logits, 1))
        indexes = list(batch_output[self.config.input_map.index])
        results = []
        if self.config.input_map.value in batch_output:
            values = batch_output[self.config.input_map.value]
        else:
            values = [0.0] * len(indexes)
        for i, (one_logits, index, value) in enumerate(zip(logits, indexes, values)):
            one_ins = {}
            one_origin = self._get_origin_row(origin_data, index)
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
        """calc the scores use the predicts or list_batch_outputs

        Args:
            predicts: list of predicts
            stage: train/test/etc.
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
