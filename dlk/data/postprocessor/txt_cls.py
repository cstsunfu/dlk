# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import pickle as pkl
from typing import Any, Dict, List, Union

import numpy as np
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


@cregister("postprocessor", "txt_cls")
class TxtClsPostProcessorConfig(BasePostProcessorConfig):
    """postprocess for text classfication"""

    label_vocab = StrField(
        value=MISSING, help="the label vocab file name, relative `meta_dir`"
    )

    class InputMap:
        logits = StrField(value="logits", help="the output logits")
        label_ids = StrField(value="label_ids", help="the label ids of the sample")
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
    top_k = IntField(
        value=0,
        minimum=0,
        help="the top k result, if set to `0` will get all the values",
    )


@register("postprocessor", "txt_cls")
class TxtClsPostProcessor(BasePostProcessor):
    """postprocess for text classfication"""

    def __init__(self, config: TxtClsPostProcessorConfig):
        super(TxtClsPostProcessor, self).__init__(config)
        self.config = config
        self.label_vocab = Vocabulary.load_from_file(
            os.path.join(self.config.meta_dir, self.config.label_vocab)
        )
        self.top_k = config.top_k if config.top_k > 0 else self.label_vocab.word_num

    def _get_origin_data(self, one_origin: pd.Series) -> Dict:
        """

        Args:
            one_origin: the original data

        Returns:
            the gather origin data

        """
        origin = {}
        if self.config.data_type == "single":
            sentence = one_origin[self.config.origin_input_map.sentence]
            origin["sentence"] = sentence
        else:
            sentence_a = one_origin[self.config.origin_input_map.sentence_a]
            origin["sentence_a"] = sentence_a
            sentence_b = one_origin[self.config.origin_input_map.sentence_b]
            origin["sentence_b"] = sentence_b
        if "labels" in one_origin:
            origin["labels"] = one_origin["labels"]
        origin["uuid"] = one_origin[self.config.origin_input_map.uuid]
        return origin

    def wrap_predict_one_batch(
        self,
        stage: str,
        batch_output: Dict,
        origin_data: pd.DataFrame,
        rt_config: Dict,
    ) -> List:
        """Process the model predict to human readable format

        Args:
            stage: train/test/etc.
            batch_output: model outputs
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
            the predicts

        """
        batch_output[self.config.input_map.logits] = (
            batch_output[self.config.input_map.logits].detach().cpu().numpy()
        )
        return self.predict_one_batch(stage, batch_output, origin_data, rt_config)

    def predict_one_batch(
        self, stage, batch_output: Dict, origin_data: pd.DataFrame, rt_config
    ) -> List:
        """Process the model predict to human readable format for one batch
        Args:
            stage: train/test/etc.
            batch_output: a dict of outputs
            origin_data: the origin pd.DataFrame data, there are some data not be able to convert to tensor
        Returns:
            the predicts of one batch
        """
        results = []
        logits = batch_output[self.config.input_map.logits]
        assert len(logits.shape) == 2
        indexes = list(batch_output[self.config.input_map.index])

        if self.config.input_map.label_ids in batch_output:
            label_ids = batch_output[self.config.input_map.label_ids]
        else:
            label_ids = [None] * len(indexes)
        for i, (one_logits, index, label_id) in enumerate(
            zip(logits, indexes, label_ids)
        ):
            one_ins = self._get_origin_data(origin_data.iloc[int(index)])

            max_val = np.max(one_logits)  # 防止数值溢出
            exp_logits = np.exp(one_logits - max_val)
            softmax_logits = exp_logits / np.sum(exp_logits)

            label_indeies = np.argsort(-softmax_logits)[: self.top_k]
            label_values = softmax_logits[label_indeies]

            predict = {}
            for i, (label_value, label_index) in enumerate(
                zip(label_values, label_indeies)
            ):
                label_name = self.label_vocab.get_word(int(label_index))
                predict[i] = [label_name, float(label_value)]
            one_ins["predicts"] = predict
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
        right_num = 0
        for one_ins in predicts:
            labels = one_ins["labels"]
            assert (
                len(labels) == 1
            ), "We currently is not support multi label in classification postprocess"
            label = labels[0]
            one_predicts = one_ins["predicts"]
            predict_label, predict_value = one_predicts[0]  # the first predict
            if label == predict_label:
                right_num += 1
        real_name = self.loss_name_map(stage)
        return {f"{real_name}_acc": right_num / len(predicts)}
