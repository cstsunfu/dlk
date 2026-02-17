# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
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


@cregister("postprocessor", "span_relation")
class SpanRelationPostProcessorConfig(BasePostProcessorConfig):
    """span based relation extraction postprocessor"""

    ignore_position = BoolField(
        value=False, help="whether to ignore the position of the entity"
    )
    ignore_char = ListField(
        value=[],
        suggestions=[[" ", "(", ")", "[", "]", "-", ".", ",", ":", "'", '"']],
        help="ignore the provided char if these char is prefix or suffix of the entity",
    )
    entity_label_vocab = StrField(
        value=MISSING,
        help="the label vocab file path of entity, it should be the same as file in the preprocessor",
    )
    relation_label_vocab = StrField(
        value=MISSING,
        help="the label vocab file path of relation, it should be the same as file in the preprocessor",
    )

    class InputMap:
        logits = StrField(value="logits", help="the entity logits")
        head_logits = StrField(
            value="head_logits", help="the head2head relation logits"
        )
        tail_logits = StrField(
            value="tail_logits", help="the tail2tail relation logits"
        )
        index = StrField(value="_index", help="the index of the sample")

    input_map = NestField(
        value=InputMap,
        help="the input map of the processor",
    )

    class OriginInputMap:
        uuid = StrField(value="uuid", help="the uuid or the id of the sample")
        sentence = StrField(value="sentence", help="the sentence of the sample")
        input_ids = StrField(value="input_ids", help="the input ids of the sample")
        entities_info = StrField(value="entities_info", help="the entities info")
        relations_info = StrField(value="relations_info", help="the relations info")
        offsets = StrField(value="offsets", help="the offsets of the tokens")
        special_tokens_mask = StrField(
            value="special_tokens_mask", help="the special tokens mask"
        )
        word_ids = StrField(value="word_ids", help="the word ids")

    origin_input_map = NestField(
        value=OriginInputMap,
        help="the origin input map",
    )
    unrelated_entity = BoolField(
        value=True, help="whether to save the unrelated entity(no pair relation)"
    )
    relation_groups = IntField(
        value=1,
        help="the relation groups",
    )
    sym = BoolField(
        value=True,
        help="whether the relation is sym",
    )
    ignore_labels = ListField(
        value=["O", "X", "S", "E"],
        help="the ignore labels",
    )
    ignore_relations = ListField(value=[], help="the ignore relations")
    entity_threshold = FloatField(
        value=0.0,
        help="the threshold of the entity logits",
    )
    relation_threshold = FloatField(
        value=0.0,
        help="the threshold of the relation logits",
    )


@register("postprocessor", "span_relation")
class SpanRelationPostProcessor(BasePostProcessor):
    """PostProcess for relation extraction task"""

    def __init__(self, config: SpanRelationPostProcessorConfig):

        super(SpanRelationPostProcessor, self).__init__(config)
        self.config = config

        self.entity_label_vocab = Vocabulary.load_from_file(
            os.path.join(self.config.meta_dir, self.config.entity_label_vocab)
        )
        self.relation_label_vocab = Vocabulary.load_from_file(
            os.path.join(self.config.meta_dir, self.config.relation_label_vocab)
        )

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
        batch_output[self.config.input_map.head_logits] = (
            batch_output[self.config.input_map.head_logits].float().cpu().numpy()
        )
        batch_output[self.config.input_map.tail_logits] = (
            batch_output[self.config.input_map.tail_logits].float().cpu().numpy()
        )
        return self.predict_one_batch(stage, batch_output, origin_data, rt_config)

    def predict_one_batch(
        self, stage, batch_output: Dict, origin_data: Any, rt_config
    ) -> List:
        batch_logits = batch_output[self.config.input_map.logits]
        batch_head_logits = batch_output[self.config.input_map.head_logits]
        batch_tail_logits = batch_output[self.config.input_map.tail_logits]
        indexes = batch_output[self.config.input_map.index]
        predicts = []
        for i, (logits, head_logits, tail_logits, index) in enumerate(
            zip(batch_logits, batch_head_logits, batch_tail_logits, indexes)
        ):
            one_ins_info = self._process4predict(
                logits, head_logits, tail_logits, index, origin_data
            )
            one_ins_info["predict_extend_return"] = self.gather_predict_extend_data(
                batch_output, i, self.config.predict_extend_return
            )
            predicts.append(one_ins_info)
        return predicts

    def _process4predict(
        self,
        logits,
        head_logits,
        tail_logits,
        index: int,
        origin_data: Any,
    ) -> Dict:

        def _get_entity_info(
            sub_tokens_index: List, offset_mapping: List, word_ids: List, label: str
        ) -> Dict:
            if not sub_tokens_index or not label:
                return {}
            start = offset_mapping[sub_tokens_index[0]][0]
            end = offset_mapping[sub_tokens_index[-1]][1]
            return {
                "start": start,
                "end": end,
                "labels": [label],
                "sub_token_start": sub_tokens_index[0],
                "sub_token_end": sub_tokens_index[1],
            }

        one_ins = {}
        origin_ins = self._get_origin_row(origin_data, index)
        one_ins["sentence"] = origin_ins[self.config.origin_input_map.sentence]
        one_ins["uuid"] = origin_ins[self.config.origin_input_map.uuid]
        one_ins["entities_info"] = origin_ins.get(
            self.config.origin_input_map.entities_info, []
        )
        one_ins["relations_info"] = origin_ins.get(
            self.config.origin_input_map.relations_info, []
        )

        word_ids = origin_ins[self.config.origin_input_map.word_ids]
        rel_token_len = len(word_ids)
        offset_mapping = origin_ins[self.config.origin_input_map.offsets][
            :rel_token_len
        ]

        predict_entities_id_info_map = {}
        entities_in_relations_id = set()
        max_entities = rel_token_len

        for label_id, start, end in (
            np.argwhere(
                logits[:, :rel_token_len, :rel_token_len] > self.config.entity_threshold
            )
            .astype(int)
            .tolist()
        ):
            predict_entity_label = self.entity_label_vocab[label_id]
            entity_info = _get_entity_info(
                [start, end], offset_mapping, word_ids, predict_entity_label
            )
            if entity_info:
                entity_id = str(uuid.uuid4())
                predict_entities_id_info_map[entity_id] = entity_info
            if len(predict_entities_id_info_map) > max_entities:
                predict_entities_id_info_map = {}
                entities_in_relations_id = set()
                break

        predict_relations_info = []
        entity_ids = list(predict_entities_id_info_map.keys())
        for from_entity_id in entity_ids:
            for to_entity_id in entity_ids:
                from_entity_info = predict_entities_id_info_map[from_entity_id]
                from_h, from_t = (
                    from_entity_info["sub_token_start"],
                    from_entity_info["sub_token_end"],
                )
                to_entity_info = predict_entities_id_info_map[to_entity_id]
                to_h, to_t = (
                    to_entity_info["sub_token_start"],
                    to_entity_info["sub_token_end"],
                )

                p1s = np.where(
                    head_logits[:, from_h, to_h] > self.config.relation_threshold
                )[0]
                p2s = np.where(
                    tail_logits[:, from_t, to_t] > self.config.relation_threshold
                )[0]
                ps = set(p1s) & set(p2s)
                labels = []
                for p in ps:
                    labels.append(self.relation_label_vocab[p])
                if labels:
                    predict_relation_info = {
                        "from": from_entity_id,
                        "to": to_entity_id,
                        "labels": labels,
                    }
                    entities_in_relations_id.add(from_entity_id)
                    entities_in_relations_id.add(to_entity_id)
                    predict_relations_info.append(predict_relation_info)
        entity_ids = entities_in_relations_id
        if self.config.unrelated_entity:
            entity_ids = predict_entities_id_info_map.keys()
        predict_entities_info = []
        for entity_id in entity_ids:
            entity_info = predict_entities_id_info_map[entity_id]
            entity_info["entity_id"] = entity_id
            predict_entities_info.append(entity_info)

        one_ins["predict_entities_info"] = predict_entities_info
        one_ins["predict_relations_info"] = predict_relations_info
        return one_ins

    def do_calc_metrics(
        self,
        predicts: List,
        stage: str,
        rt_config: Dict,
    ) -> Dict:
        real_name = self.loss_name_map(stage)
        entity_precision, entity_recall, entity_f1 = self._do_calc_entity_metrics(
            predicts
        )
        (
            relation_precision,
            relation_recall,
            relation_f1,
        ) = self._do_calc_relation_metrics(predicts)
        return {
            f"{real_name}_ent_p": entity_precision * 100,
            f"{real_name}_ent_r": entity_recall * 100,
            f"{real_name}_ent_f1": entity_f1 * 100,
            f"{real_name}_rel_p": relation_precision * 100,
            f"{real_name}_rel_r": relation_recall * 100,
            f"{real_name}_rel_f1": relation_f1 * 100,
        }

    def _do_calc_relation_metrics(self, predicts: List):
        def _group_relations_info(
            relations_info: List[Dict], entities_info: List[Dict], text
        ) -> Dict[str, Set[Tuple]]:
            def _norm_entity(entity):
                start_position, end_position = entity["start"], entity["end"]
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
                    return entity["start"], entity["end"]
                return start_position, end_position

            entities_id_info_map = {}
            for entity_info in entities_info:
                entities_id_info_map[entity_info["entity_id"]] = entity_info

            flat_relations = {}
            for relation_info in relations_info:
                from_entity = entities_id_info_map[relation_info["from"]]
                to_entity = entities_id_info_map[relation_info["to"]]
                relation_type = relation_info["labels"][0]
                if relation_type not in flat_relations:
                    flat_relations[relation_type] = set()
                from_start, from_end = _norm_entity(from_entity)
                to_start, to_end = _norm_entity(to_entity)

                flat_relations[relation_type].add(
                    (from_start, from_end, to_start, to_end)
                )
            return flat_relations

        relation_match_info = {}
        for predict in predicts:
            text = predict["sentence"]
            predict_relations = _group_relations_info(
                predict["predict_relations_info"],
                predict["predict_entities_info"],
                text,
            )
            ground_truth_relations = _group_relations_info(
                predict["relations_info"], predict["entities_info"], text
            )
            for key in set(ground_truth_relations.keys()).union(
                set(predict_relations.keys())
            ):
                if key not in relation_match_info:
                    relation_match_info[key] = {
                        "match": 0,
                        "miss": 0,
                        "wrong": 0,
                    }
            for key in ground_truth_relations:
                for ground_truth_relation in ground_truth_relations[key]:
                    if ground_truth_relation in predict_relations.get(key, {}):
                        predict_relations[key].remove(ground_truth_relation)
                        relation_match_info[key]["match"] += 1
                    elif self.config.sym and (
                        ground_truth_relation[2],
                        ground_truth_relation[3],
                        ground_truth_relation[0],
                        ground_truth_relation[1],
                    ) in predict_relations.get(key, {}):
                        predict_relations[key].remove(
                            (
                                ground_truth_relation[2],
                                ground_truth_relation[3],
                                ground_truth_relation[0],
                                ground_truth_relation[1],
                            )
                        )
                        relation_match_info[key]["match"] += 1
                    else:
                        relation_match_info[key]["miss"] += 1
            for key in predict_relations:
                for wrong in predict_relations[key]:
                    relation_match_info[key]["wrong"] += 1
        all_tp, all_fn, all_fp = 0, 0, 0

        def _care_div(a, b):
            if b == 0:
                return 0.0
            return a / b

        category_data = []
        for key in relation_match_info:
            tp = relation_match_info[key]["match"]
            fn = relation_match_info[key]["miss"]
            fp = relation_match_info[key]["wrong"]
            precision = _care_div(tp, tp + fp)
            recall = _care_div(tp, tp + fn)
            f1 = 2 * _care_div(precision * recall, precision + recall)

            all_tp += tp
            all_fn += fn
            all_fp += fp
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
        f1 = 2 * _care_div(precision * recall, precision + recall)
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
        headers = ["Relation Type", "Precision", "Recall", "F1", "TP", "FP", "FN"]
        table = tabulate(category_data, headers, tablefmt="rounded_grid")

        logger.info("\nRelation Metrics:\n" + table)

        return precision, recall, f1

    def _do_calc_entity_metrics(self, predicts: List):
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

        def _calc_entity_score(predict_list: List, ground_truth_list: List):
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

            all_tp, all_fn, all_fp = 0, 0, 0

            category_data = []
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

        entity_precision, entity_recall, entity_f1 = _calc_entity_score(
            all_predicts, all_ground_truths
        )
        return entity_precision, entity_recall, entity_f1
