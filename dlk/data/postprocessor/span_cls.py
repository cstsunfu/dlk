# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import pickle as pkl
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from intc import (
    MISSING,
    Base,
    BoolField,
    FloatField,
    IntField,
    ListField,
    NestField,
    StrField,
    cregister,
)
from tabulate import tabulate

from dlk.data.postprocessor import BasePostProcessor, BasePostProcessorConfig
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
    threshold = FloatField(
        value=0.0,
        help="the threshold of the logits, if the logits is less than this value, we will ignore this entity",
    )


@register("postprocessor", "span_cls")
class SpanClsPostProcessor(BasePostProcessor):
    """PostProcess for span classification task"""

    def __init__(self, config: SpanClsPostProcessorConfig):
        super(SpanClsPostProcessor, self).__init__(config)
        self.config = config
        self.label_vocab = Vocabulary.load_from_file(
            os.path.join(self.config.meta_dir, self.config.label_vocab)
        )

    def _process4predict(
        self, predict_logits: torch.FloatTensor, index: int, origin_data: Any
    ) -> Dict:
        def _get_entity_info(
            sub_tokens_index: List, offset_mapping: List, word_ids: List, label: str
        ) -> Dict:
            if not sub_tokens_index or not label:
                return {}
            start = offset_mapping[sub_tokens_index[0]][0]
            end = offset_mapping[sub_tokens_index[-1]][1]
            return {"start": start, "end": end, "labels": [label]}

        one_ins = {}
        origin_ins = self._get_origin_row(origin_data, index)
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

        max_entities = rel_token_len
        for label_id, start, end in (
            np.argwhere(
                predict_logits[:, :rel_token_len, :rel_token_len]
                > self.config.threshold
            )
            .astype(int)
            .tolist()
        ):
            predict_label = self.label_vocab[label_id]
            entity_info = _get_entity_info(
                [start, end], offset_mapping, word_ids, predict_label
            )
            if entity_info:
                predict_entities_info.append(entity_info)

            if len(predict_entities_info) > max_entities:
                break

        one_ins["predict_entities_info"] = predict_entities_info
        return one_ins

    def wrap_predict_one_batch(
        self,
        stage: str,
        batch_output: Dict,
        origin_data: Any,
        rt_config: Dict,
    ) -> List:
        batch_output[self.config.input_map.logits] = (
            batch_output[self.config.input_map.logits].float().cpu().numpy()
        )
        return self.predict_one_batch(stage, batch_output, origin_data, rt_config)

    def predict_one_batch(
        self, stage, batch_output: Dict, origin_data: Any, rt_config
    ) -> List:
        batch_logits = batch_output[self.config.input_map.logits]
        indexes = batch_output[self.config.input_map.index]
        predicts = []
        for i, (predict, index) in enumerate(zip(batch_logits, indexes)):
            one_ins = self._process4predict(predict, index, origin_data)
            one_ins["predict_extend_return"] = self.gather_predict_extend_data(
                batch_output, i, self.config.predict_extend_return
            )
            predicts.append(one_ins)
        return predicts

    def do_calc_metrics(
        self,
        predicts: List,
        stage: str,
        rt_config: Dict,
    ) -> Dict:
        def _group_entities_info(entities_info: List[Dict], text: str) -> Dict:
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
                if start_position == end_position:
                    start_position, end_position = item["start"], item["end"]

                if self.config.ignore_position:
                    info[label].append(text[item["start"] : item["end"]].strip())
                else:
                    info[label].append((start_position, end_position))
            return info

        def _calc_score(predict_list: List, ground_truth_list: List):
            category_tp = {}
            category_fp = {}
            category_fn = {}

            def _care_div(a, b):
                if b == 0:
                    return 0.0
                return a / b

            def _calc_num(_pred: List, _ground_truth: List):
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

            category_data = []
            all_tp, all_fn, all_fp = 0, 0, 0
            for key in category_tp:
                tp, fn, fp = category_tp[key], category_fn[key], category_fp[key]
                all_tp += tp
                all_fn += fn
                all_fp += fp
                precision = _care_div(tp, tp + fp)
                recall = _care_div(tp, tp + fn)
                f1 = _care_div(2 * precision * recall, precision + recall)

                category_data.append(
                    [
                        key,
                        f"{precision*100:.2f}%",
                        f"{recall*100:.2f}%",
                        f"{f1 * 100:.2f}%",
                        tp,
                        fp,
                        fn,
                    ]
                )
            precision = _care_div(all_tp, all_tp + all_fp)
            recall = _care_div(all_tp, all_tp + all_fn)
            f1 = _care_div(2 * precision * recall, precision + recall)

            category_data.append(
                [
                    "Overall",
                    f"{precision*100:.2f}%",
                    f"{recall*100:.2f}%",
                    f"{f1 * 100:.2f}%",
                    all_tp,
                    all_fp,
                    all_fn,
                ]
            )

            headers = ["Entity Type", "Precision", "Recall", "F1", "TP", "FP", "FN"]
            table = tabulate(category_data, headers, tablefmt="rounded_grid")

            logger.info("\nEntity Metrics:\n" + table)
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
        return {
            f"{real_name}_precision": precision * 100,
            f"{real_name}_recall": recall * 100,
            f"{real_name}_f1": f1 * 100,
        }
