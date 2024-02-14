# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import pickle as pkl
import uuid
from typing import Dict, List, Optional, Set, Tuple, Union

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
    tokenizer_path = StrField(
        value=MISSING, help="the tokenizer file path, is not effected by `meta_dir`"
    )

    class InputMap:
        entity_logits = StrField(value="entity_logits", help="the entity logits")
        relation_logits = StrField(value="relation_logits", help="the relation logits")
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
        relations_info = StrField(value="relations_info", help="the relations info")
        offsets = StrField(value="offsets", help="the offsets of the tokens")
        special_tokens_mask = StrField(
            value="special_tokens_mask", help="the special tokens mask"
        )
        word_ids = StrField(value="word_ids", help="the word ids")

    origin_input_map = NestField(
        value=OriginInputMap,
        help="the origin input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )
    unrelated_entity = BoolField(
        value=True, help="whether to save the unrelated entity(no pair relation)"
    )
    relation_groups = IntField(
        value=1,
        help="the relation groups, when set label_seperate==True in span_relation_relabel, the relation_groups maybe >1, but currently is not Tested",
    )
    sym = BoolField(
        value=True,
        help="whether the relation is sym, if sym==True, we can just calc the upper trim and set down trim as -100 to ignore",
    )
    ignore_labels = ListField(
        value=["O", "X", "S", "E"],
        help="the ignore labels, if the entity label in this list, we will ignore this entity",
    )
    ignore_relations = ListField(value=[], help="the ignore relations")


@register("postprocessor", "span_relation")
class SpanRelationPostProcessor(BasePostProcessor):
    """PostProcess for sequence labeling task"""

    def __init__(self, config: SpanRelationPostProcessorConfig):
        super(SpanRelationPostProcessor, self).__init__(config)
        self.config = config

        self.entity_label_vocab = Vocabulary.load_from_file(
            os.path.join(self.config.meta_dir, self.config.entity_label_vocab)
        )
        self.relation_label_vocab = Vocabulary.load_from_file(
            os.path.join(self.config.meta_dir, self.config.relation_label_vocab)
        )
        with open(self.config.tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_str = json.dumps(json.load(f))
        self.tokenizer = Tokenizer.from_str(tokenizer_str)

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
            batch_entity_logits = outputs[self.config.input_map.entity_logits]
            batch_relation_logits = outputs[self.config.input_map.relation_logits]

            indexes = list(outputs[self.config.input_map.index])
            for entity_logits, relation_logits, index in zip(
                batch_entity_logits, batch_relation_logits, indexes
            ):
                one_ins_info = self._process4predict(
                    entity_logits, relation_logits, index, origin_data
                )
                predicts.append(one_ins_info)
        return predicts

    def _process4predict(
        self,
        entity_logits: torch.FloatTensor,
        relation_logits: torch.FloatTensor,
        index: int,
        origin_data: pd.DataFrame,
    ) -> Dict:
        """gather the predict and origin text and ground_truth_entities_info for predict

        Args:
            entity_logits: the predict entity span logits
            relation_logits: the predict relation logits
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
            return {
                "start": start,
                "end": end,
                "labels": [label],
                "sub_token_start": sub_tokens_index[0],
                "sub_token_end": sub_tokens_index[1],
            }

        one_ins = {}
        origin_ins = origin_data.iloc[int(index)]
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
        predict_entity_ids = entity_logits.argmax(-1).cpu().numpy()

        # gather all predicted entities
        entity_set_d = (
            {}
        )  # set_d, set_e, set_t defined in https://arxiv.org/pdf/2010.13415.pdf Algorithm 1
        entity_cnt = 0
        for i in range(rel_token_len):
            for j in range(i, rel_token_len):
                if word_ids[i] is None or word_ids[j] is None:
                    continue
                predict_entity_id = predict_entity_ids[i][j]
                predict_entity = self.entity_label_vocab[predict_entity_id]
                if predict_entity in {
                    self.entity_label_vocab.pad,
                    self.entity_label_vocab.unknown,
                }:
                    continue
                else:
                    entity_info = _get_entity_info(
                        [i, j], offset_mapping, word_ids, predict_entity
                    )
                    if entity_info:
                        entity_id = str(uuid.uuid1())
                        predict_entities_id_info_map[entity_id] = entity_info
                        entity_set_d[i] = entity_set_d.get(i, [])
                        entity_set_d[i].append((i, j, entity_id))
                        entity_cnt += 1
            if entity_cnt > rel_token_len:
                # HACK: too many predict, maybe wrong
                break

        # all predicted relations entity tail pair
        entity_tail_pair_set_e = (
            set()
        )  # for each element (first_entity_tail_token_idx, second_entity_tail_token_idx, relation_idx)
        entity_tail_pair_set_e_with_relation_id = (
            set()
        )  # for each element (first_entity_tail_token_idx, second_entity_tail_token_idx, relation_idx, tail_relation_label_id)

        # all candidate relation set(only consider the entity head pair and the entity_set_d)
        candidate_relation_set_c = (
            set()
        )  # for each element (first_entity_info, second_entity_info, relation_idx, head_relation_label_id)
        candidate_cnt = 0
        for relation_idx in range(self.config.relation_groups):
            head_to_head_logits = relation_logits[relation_idx * 2]
            tail_to_tail_logits = relation_logits[relation_idx * 2 + 1]
            head_to_head_ids = head_to_head_logits.argmax(-1).cpu().numpy()
            tail_to_tail_ids = tail_to_tail_logits.argmax(-1).cpu().numpy()
            for i in range(rel_token_len):
                if candidate_cnt > 2 * rel_token_len:
                    # HACK: too many predict, maybe wrong
                    break
                for j in range(i if self.config.sym else 0, rel_token_len):
                    if word_ids[i] is None or word_ids[j] is None:
                        continue
                    predict_tail_id = tail_to_tail_ids[i][j]
                    predict_tail_relation = self.config.relation_label_vocab[
                        predict_tail_id
                    ]
                    if predict_tail_relation not in {
                        self.relation_label_vocab.pad,
                        self.relation_label_vocab.unknown,
                    }:
                        entity_tail_pair_set_e.add((i, j, relation_idx))
                        entity_tail_pair_set_e_with_relation_id.add(
                            (i, j, relation_idx, predict_tail_id)
                        )

                    predict_head_id = head_to_head_ids[i][j]
                    predict_head_relation = self.relation_label_vocab[predict_head_id]
                    if predict_head_relation not in {
                        self.relation_label_vocab.pad,
                        self.relation_label_vocab.unknown,
                    }:
                        for first_entity in entity_set_d.get(i, []):
                            for second_entity in entity_set_d.get(j, []):
                                candidate_relation_set_c.add(
                                    (
                                        first_entity,
                                        second_entity,
                                        relation_idx,
                                        predict_head_id,
                                    )
                                )
                                candidate_cnt += 1
        predict_relations_info = []
        for candidate_relation in candidate_relation_set_c:
            (
                first_entity,
                second_entity,
                relation_idx,
                predict_head_id,
            ) = candidate_relation
            if self.config.sym and first_entity[1] > second_entity[1]:
                first_entity, second_entity = second_entity, first_entity
            if (
                first_entity[1],
                second_entity[1],
                relation_idx,
            ) in entity_tail_pair_set_e:
                # HACK: if (first_entity[1], second_entity[1], relation_idx, predict_head_id) not in entity_tail_pair_set_e_with_relation_id, we can do better on it
                predict_label = self.relation_label_vocab[predict_head_id]

                if first_entity[2] not in entities_in_relations_id:
                    entities_in_relations_id.add(first_entity[2])
                if second_entity[2] not in entities_in_relations_id:
                    entities_in_relations_id.add(second_entity[2])
                predict_relation_info = {
                    "from": first_entity[2],
                    "to": second_entity[2],
                    "labels": [predict_label],
                }
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
        real_name = self.loss_name_map(stage)
        entity_precision, entity_recall, entity_f1 = self._do_calc_entity_metrics(
            predicts, list_batch_outputs
        )
        (
            relation_precision,
            relation_recall,
            relation_f1,
        ) = self._do_calc_relation_metrics(predicts, list_batch_outputs)
        logger.info(
            f"{real_name:>8}_entity_precision: {entity_precision*100:.2f}, {real_name:>8}_recall: {entity_recall*100:.2f}, {real_name:>8}_f1: {entity_f1*100:.2f}"
        )
        logger.info(
            f"{real_name:>8}_relation_precision: {relation_precision*100:.2f}, {real_name:>8}_recall: {relation_recall*100:.2f}, {real_name:>8}_f1: {relation_f1*100:.2f}"
        )
        return {
            f"{real_name}_ent_p": entity_precision * 100,
            f"{real_name}_ent_r": entity_recall * 100,
            f"{real_name}_ent_f1": entity_f1 * 100,
            f"{real_name}_rel_p": relation_precision * 100,
            f"{real_name}_rel_r": relation_recall * 100,
            f"{real_name}_rel_f1": relation_f1 * 100,
        }

    def _do_calc_relation_metrics(self, predicts: List, list_batch_outputs: List[Dict]):
        """calc relation related metrics
        Returns:
            relation related metrics

        """

        def _group_relations_info(
            relations_info: List[Dict], entities_info: List[Dict]
        ) -> Dict[str, Set[Tuple]]:
            """flat the relations info by relations info dict and entities info dict

            Args:
                relations_info (List[Dict]): like
                        [
                            {
                                "labels": [
                                    "belong_to"
                                ],
                                "from": "bd7a2928-52a8-11ed-8b5c-18c04d299e80", # id of entity
                                "to": "bd798da3-52a8-11ed-801c-18c04d299e80",
                            }
                        ]
                entities_info (List[Dict]): like
                        [
                            {
                                "entity_id": "bd798da3-52a8-11ed-801c-18c04d299e80",
                                "start_sub_token":1,
                                "end_sub_token": 3,
                                "labels": [
                                    "Brand"
                                ]
                            },
                            {
                                "entity_id": "bd7a2928-52a8-11ed-8b5c-18c04d299e80",
                                "start_sub_token":5,
                                "end_sub_token": 9,
                                "labels": [
                                    "Product"
                                ]
                            }
                        ],

            Returns:
                relations type relation info set pair, like
                    {
                        "belong_to": {
                            (1, 3, 5, 9),
                            ...
                        }
                    }

            Explanation:
                in the returns example,
                    1 ==> from_entity.sub_token_start
                    3 ==> from_entity.sub_token_end
                    5 ==> to_entity.sub_token_start
                    9 ==> to_entity.sub_token_end
            belong_to ==> the relation type
            """

            def _norm_entity(entity):
                """norm the entity start and end by the ignore_char

                Args:
                    entity: one entity is like:
                        {
                            "start": .,
                            "end": .,
                            "sub_token_start": .,
                            "sub_token_end": .,
                        }
                Returns: norm entity
                """
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
                if (
                    start_position == end_position
                ):  # if the entity after remove ignore char be null, we set it to origin
                    return entity["start"], entity["end"]
                return start_position, end_position

            entities_id_info_map = {}
            for entity_info in entities_info:
                entities_id_info_map[entity_info["entity_id"]] = entity_info

            flat_relations = {}
            for relation_info in relations_info:
                from_entity = entities_id_info_map[relation_info["from"]]
                to_entity = entities_id_info_map[relation_info["to"]]
                relation_type = relation_info["labels"][
                    0
                ]  # NOTE: you can also change the relation type format for different label format
                if relation_type not in flat_relations:
                    flat_relations[relation_type] = set()
                from_start, from_end = _norm_entity(from_entity)
                to_start, to_end = _norm_entity(to_entity)

                # flat_relations[relation_type].add((from_entity['sub_token_start'], from_entity['sub_token_end'], to_entity['sub_token_start'], to_entity['sub_token_end']))
                flat_relations[relation_type].add(
                    (from_start, from_end, to_start, to_end)
                )
            return flat_relations

        relation_match_info = {}
        for predict in predicts:
            text = predict["sentence"]
            predict_relations = _group_relations_info(
                predict["predict_relations_info"], predict["predict_entities_info"]
            )
            ground_truth_relations = _group_relations_info(
                predict["relations_info"], predict["entities_info"]
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
            """return a/b or 0.0 if b == 0"""
            if b == 0:
                return 0.0
            return a / b

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
            logger.info(
                f"For {'relation':16} 「{key[:16]:16}」, the precision={precision*100 :.2f}%, the recall={recall*100:.2f}%, f1={f1*100:.2f}%"
            )

        precision = _care_div(all_tp, all_tp + all_fp)
        recall = _care_div(all_tp, all_tp + all_fn)
        f1 = 2 * _care_div(precision * recall, precision + recall)
        return precision, recall, f1

    def _do_calc_entity_metrics(self, predicts: List, list_batch_outputs: List[Dict]):
        """calc entity related metrics
        Returns:
            entity related metrics

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

        def _calc_entity_score(predict_list: List, ground_truth_list: List):
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
                    f"For {'entity':16} 「{key[:16]:16}」, the precision={precision*100 :.2f}%, the recall={recall*100:.2f}%, f1={f1*100:.2f}%"
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

        entity_precision, entity_recall, entity_f1 = _calc_entity_score(
            all_predicts, all_ground_truths
        )
        return entity_precision, entity_recall, entity_f1

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
