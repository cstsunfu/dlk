# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import pickle as pkl
from typing import Dict, List, Optional, Tuple, Union

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
from tokenizers import Tokenizer

from dlk.data.postprocessor import BasePostProcessor, BasePostProcessorConfig
from dlk.utils.io import open
from dlk.utils.register import register
from dlk.utils.vocab import Vocabulary

logger = logging.getLogger(__name__)


@cregister("postprocessor", "span_cls")
class SpanClsPostProcessorConfig(BasePostProcessorConfig):
    """span classfication postprocessor"""

    ignore_position = BoolField(
        value=False, help="whether to ignore the position of the entity"
    )
    ignore_char = ListField(
        value=[],
        suggestions=[[" ", "(", ")", "[", "]", "-", ".", ",", ":", "'", '"']],
        help="ignore the provided char if these char is prefix or suffix of the entity",
    )
    label_vocab = StrField(value=MISSING, help="the label vocab file path")
    tokenizer_path = StrField(
        value=MISSING, help="the tokenizer file path, is not effected by `meta_dir`"
    )

    class InputMap:
        logits = StrField(value="logits", help="the output logits")
        index = StrField(value="_index", help="the index of the sample")

    input_map = NestField(
        value=InputMap,
        help="the input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )

    class OriginInputMap:
        uuid = StrField(value="uuid", help="the uuid or the id of the sample")
        sentence = StrField(value="sentence", help="the sentence of the sample")
        input_ids = StrField(value="input_ids", help="the input ids of the sample")
        entities_info = StrField(value="entities_info", help="the entities info")
        offsets = StrField(value="offsets", help="the offsets of the tokens")
        special_tokens_mask = StrField(
            value="special_tokens_mask", help="the special tokens mask"
        )
        word_ids = StrField(value="word_ids", help="the word ids")

    origin_input_map = NestField(
        value=OriginInputMap,
        help="the origin input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )
    ignore_labels = ListField(
        value=["O", "X", "S", "E"],
        help="the ignore labels, if the entity label in this list, we will ignore this entity",
    )


@register("postprocessor", "span_cls")
class SpanClsPostProcessor(BasePostProcessor):
    """PostProcess for sequence labeling task"""

    def __init__(self, config: SpanClsPostProcessorConfig):
        super(SpanClsPostProcessor, self).__init__(config)
        self.config = config
        self.label_vocab = Vocabulary.load_from_file(
            os.path.join(self.config.meta_dir, self.config.label_vocab)
        )
        with open(self.config.tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_str = json.dumps(json.load(f))
        self.tokenizer = Tokenizer.from_str(tokenizer_str)

    def _process4predict(
        self, predict_logits: torch.FloatTensor, index: int, origin_data: pd.DataFrame
    ) -> Dict:
        """gather the predict and origin text and ground_truth_entities_info for predict

        Args:
            predict: the predict span logits
            index: the data index in origin_data
            origin_data: the origin pd.DataFrame

        Returns:
            >>> one_ins info
            >>> {
            >>>     "sentence": "...",
            >>>     "uuid": "..",
            >>>     "entities_info": [".."],
            >>>     "predict_entities_info": [".."],
            >>> }

        """

        def _get_entity_info(
            sub_tokens_index: List, offset_mapping: List, word_ids: List, label: str
        ) -> Dict:
            """gather sub_tokens to get the start and end

            Args:
                sub_tokens_index: the entity tokens index list
                offset_mapping: every token offset in text
                word_ids: every token in the index of words
                label: predict label

            Returns:
                entity_info

            """
            if not sub_tokens_index or not label:
                return {}
            start = offset_mapping[sub_tokens_index[0]][0]
            end = offset_mapping[sub_tokens_index[-1]][1]
            return {"start": start, "end": end, "labels": [label]}

        one_ins = {}
        origin_ins = origin_data.iloc[int(index)]
        one_ins["sentence"] = origin_ins[self.config.origin_input_map.sentence]
        one_ins["uuid"] = origin_ins[self.config.origin_input_map.uuid]
        one_ins["entities_info"] = origin_ins[
            self.config.origin_input_map.entities_info
        ]

        word_ids = origin_ins[self.config.origin_input_map.word_ids]
        rel_token_len = len(word_ids)
        offset_mapping = origin_ins[self.config.origin_input_map.offsets][
            :rel_token_len
        ]

        predict_entities_info = []
        predict_label_ids = predict_logits.argmax(-1).cpu().numpy()
        for i in range(rel_token_len):
            for j in range(i, rel_token_len):
                if word_ids[i] is None or word_ids[j] is None:
                    continue
                predict_label_id = predict_label_ids[i][j]
                predict_label = self.label_vocab[predict_label_id]
                if (
                    predict_label == self.label_vocab.pad
                    or predict_label == self.label_vocab.unknown
                ):
                    continue
                else:
                    entity_info = _get_entity_info(
                        [i, j], offset_mapping, word_ids, predict_label
                    )
                    if entity_info:
                        predict_entities_info.append(entity_info)
        one_ins["predict_entities_info"] = predict_entities_info
        return one_ins

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
        predicts = []
        if self.config.origin_input_map.sentence not in origin_data:
            logger.error(
                f"{self.config.origin_input_map.sentence} not in the origin data"
            )
            raise PermissionError(
                f"{self.config.origin_input_map.sentence} must be provided"
            )
        if self.config.origin_input_map.uuid not in origin_data:
            logger.error(f"{self.config.origin_input_map.uuid} not in the origin data")
            raise PermissionError(
                f"{self.config.origin_input_map.uuid} must be provided"
            )

        predicts = []
        for outputs in list_batch_outputs:
            batch_logits = outputs[self.config.input_map.logits]
            # batch_special_tokens_mask = outputs[self.config.special_tokens_mask]

            indexes = list(outputs[self.config.input_map.index])

            for i, (predict, index) in enumerate(zip(batch_logits, indexes)):
                one_ins = self._process4predict(predict, index, origin_data)
                one_ins["predict_extend_return"] = self.gather_predict_extend_data(
                    outputs, i, self.config.predict_extend_return
                )
                predicts.append(one_ins)
        return predicts

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
            the named scores, recall, precision, f1

        """

        def _group_entities_info(entities_info: List[Dict], text: str) -> Dict:
            """gather the same labeled entity to the same list

            Args:
                entities_info:
                    >>> [
                    >>>     {
                    >>>         "start": start1,
                    >>>         "end": end1,
                    >>>         "labels": ["label_1"]
                    >>>     },
                    >>>     {
                    >>>         "start": start2,
                    >>>         "end": end2,
                    >>>         "labels": ["label_2"]
                    >>>     },....
                    >>> ]
                text: be labeled text

            Returns:
                >>> { "label_1" [text[start1:end1]], "label_2": [text[start_2: end_2]]...}

            """
            info = {}
            for item in entities_info:
                label = item["labels"][0]
                if label not in info:
                    info[label] = []
                start_position, end_position = item["start"], item["end"]
                while start_position < end_position:
                    if text[start_position] in self.config.ignore_char:
                        start_position += 1
                    else:
                        break
                while start_position < end_position:
                    if text[end_position - 1] in self.config.ignore_char:
                        end_position -= 1
                    else:
                        break
                if (
                    start_position == end_position
                ):  # if the entity after remove ignore char be null, we set it to origin
                    start_position, end_position = item["start"], item["end"]

                if self.config.ignore_position:
                    info[label].append(text[item["start"] : item["end"]].strip())
                else:
                    info[label].append((start_position, end_position))
            return info

        def _calc_score(predict_list: List, ground_truth_list: List):
            """use predict_list and ground_truth_list to calc scores

            Args:
                predict_list: list of predict
                ground_truth_list: list of ground_truth

            Returns:
                precision, recall, f1

            """
            category_tp = {}
            category_fp = {}
            category_fn = {}

            def _care_div(a, b):
                """return a/b or 0.0 if b == 0"""
                if b == 0:
                    return 0.0
                return a / b

            def _calc_num(_pred: List, _ground_truth: List):
                """calc tp, fn, fp

                Args:
                    pred: pred list
                    ground_truth: groud truth list

                Returns:
                    tp, fn, fp

                """
                num_p = len(_pred)
                num_t = len(_ground_truth)
                truth = 0
                for p in _pred:
                    if p in _ground_truth:
                        truth += 1
                return truth, num_t - truth, num_p - truth

            for predict, ground_truth in zip(predict_list, ground_truth_list):
                keys = set(list(predict.keys()) + list(ground_truth.keys()))
                for key in keys:
                    tp, fn, fp = _calc_num(
                        predict.get(key, []), ground_truth.get(key, [])
                    )
                    category_tp[key] = category_tp.get(key, 0) + tp
                    category_fn[key] = category_fn.get(key, 0) + fn
                    category_fp[key] = category_fp.get(key, 0) + fp

            all_tp, all_fn, all_fp = 0, 0, 0
            for key in category_tp:
                tp, fn, fp = category_tp[key], category_fn[key], category_fp[key]
                all_tp += tp
                all_fn += fn
                all_fp += fp
                precision = _care_div(tp, tp + fp)
                recall = _care_div(tp, tp + fn)
                f1 = _care_div(2 * precision * recall, precision + recall)
                logger.info(
                    f"For entity 「{key}」, the precision={precision*100 :.2f}%, the recall={recall*100:.2f}%, f1={f1*100:.2f}%"
                )
            precision = _care_div(all_tp, all_tp + all_fp)
            recall = _care_div(all_tp, all_tp + all_fn)
            f1 = _care_div(2 * precision * recall, precision + recall)
            return precision, recall, f1

        all_predicts = []
        all_ground_truths = []
        for predict in predicts:
            text = predict["sentence"]
            predict_ins = _group_entities_info(predict["predict_entities_info"], text)
            ground_truth_ins = _group_entities_info(predict["entities_info"], text)
            all_predicts.append(predict_ins)
            all_ground_truths.append(ground_truth_ins)

        precision, recall, f1 = _calc_score(all_predicts, all_ground_truths)
        real_name = self.loss_name_map(stage)
        logger.info(
            f"{real_name}_precision: {precision*100}, {real_name}_recall: {recall*100}, {real_name}_f1: {f1*100}"
        )
        return {
            f"{real_name}_precision": precision * 100,
            f"{real_name}_recall": recall * 100,
            f"{real_name}_f1": f1 * 100,
        }

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
        if (
            rt_config["current_step"] >= self.config.start_save_step
            or rt_config["current_epoch"] >= self.config.start_save_epoch
        ):
            save_condition = True
        if save_condition:
            save_dir = os.path.join(
                self.config.save_root_path, self.config.save_dir.get(stage, "")
            )
            if "current_step" in rt_config:
                save_file = os.path.join(
                    save_dir, f"step_{str(rt_config['current_step'])}_predict.json"
                )
            else:
                save_file = os.path.join(save_dir, "predict.json")
            logger.info(f"Save the {stage} predict data at {save_file}")
            with open(save_file, "w") as f:
                json.dump(predicts, f, indent=4, ensure_ascii=False)
