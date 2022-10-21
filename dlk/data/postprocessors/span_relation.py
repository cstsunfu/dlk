# Copyright cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:// www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle as pkl
import json
from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
import pandas as pd
import torch
from typing import Dict
from dlk.data.postprocessors import postprocessor_register, postprocessor_config_register, IPostProcessor, IPostProcessorConfig
from dlk.utils.logger import Logger
from dlk.utils.vocab import Vocabulary
from dlk.utils.io import open
from tokenizers import Tokenizer
import torchmetrics
import uuid
logger = Logger.get_logger()


@postprocessor_config_register('span_relation')
class SpanRelationPostProcessorConfig(IPostProcessorConfig):
    default_config = {
            "_name": "span_relation",
            "config": {
                "meta": "*@*",
                "ignore_position": False, # calc the metrics, whether ignore the ground_truth and predict position info.( if set to true, only focus on the entity content not position.)
                "ignore_char": " ", # if the entity begin or end with this char, will ignore these char
                # "ignore_char": " ()[]-.,:", # if the entity begin or end with this char, will ignore these char
                "meta_data": {
                    "entity_label_vocab": 'entity_label_vocab',
                    "relation_label_vocab": 'relation_label_vocab',
                    "tokenizer": "tokenizer",
                },
                "input_map": {
                    "entity_logits": "entity_logits",
                    "relation_logits": "relation_logits",
                    "_index": "_index",
                },
                "origin_input_map": {
                    "uuid": "uuid",
                    "sentence": "sentence",
                    "input_ids": "input_ids",
                    "entities_info": "entities_info",
                    "relations_info": "relations_info",
                    "offsets": "offsets",
                    "special_tokens_mask": "special_tokens_mask",
                    "word_ids": "word_ids",
                    "label_ids": "label_ids",
                },
                "save_root_path": ".",  # save data root dir
                "save_path": {
                    "valid": "valid",  # relative dir for valid stage
                    "test": "test",    # relative dir for test stage
                },
                "relation_types": 1,  # relation type numbers
                "sym": True, # whther the from entity and end entity can swap in relations( if sym==True, we can just calc the upper trim and set down trim as -100 to ignore)
                "start_save_step": 0,  # -1 means the last
                "start_save_epoch": -1,
                "ignore_relations": [],
                "ignore_labels": [], # Out, Out, Start, End
            }
        }
    """Config for SpanRelationPostProcessor

    Config Example:
    """

    def __init__(self, config: Dict):
        super(SpanRelationPostProcessorConfig, self).__init__(config)

        self.ignore_labels = set(self.config['ignore_labels'])
        self.ignore_relations = set(self.config['ignore_relations'])
        self.ignore_char = set(self.config['ignore_char'])
        self.ignore_position = self.config['ignore_position']
        self.relation_types = self.config['relation_types']
        self.sym = self.config['sym']

        self.sentence = self.origin_input_map['sentence']
        self.offsets = self.origin_input_map['offsets']
        self.entities_info = self.origin_input_map['entities_info']
        self.relations_info = self.origin_input_map['relations_info']
        self.uuid = self.origin_input_map['uuid']
        self.word_ids = self.origin_input_map['word_ids']
        self.special_tokens_mask = self.origin_input_map['special_tokens_mask']
        self.input_ids = self.origin_input_map['input_ids']
        self.label_ids = self.origin_input_map['label_ids']

        self.entity_logits = self.input_map['entity_logits']
        self.relation_logits = self.input_map['relation_logits']
        self._index = self.input_map['_index']

        if isinstance(self.config['meta'], str):
            with open(self.config['meta'], 'rb') as f:
                meta = pkl.load(f)
        else:
            raise PermissionError("You must provide meta data(vocab & tokenizer) for ner postprocess.")

        entity_vocab_trace_path = []
        entity_vocab_trace_path_str = self.config['meta_data']['entity_label_vocab']
        if entity_vocab_trace_path_str and entity_vocab_trace_path_str.strip()!='.':
            entity_vocab_trace_path = entity_vocab_trace_path_str.split('.')
        assert entity_vocab_trace_path, "We need vocab and tokenizer all in meta, so you must provide the trace path from meta"
        self.entity_label_vocab = meta
        for trace in entity_vocab_trace_path:
            self.entity_label_vocab = self.entity_label_vocab[trace]
        self.entity_label_vocab = Vocabulary.load(self.entity_label_vocab)

        relation_vocab_trace_path = []
        relation_vocab_trace_path_str = self.config['meta_data']['relation_label_vocab']
        if relation_vocab_trace_path_str and relation_vocab_trace_path_str.strip()!='.':
            relation_vocab_trace_path = relation_vocab_trace_path_str.split('.')
        assert relation_vocab_trace_path, "We need vocab and tokenizer all in meta, so you must provide the trace path from meta"
        self.relation_label_vocab = meta
        for trace in relation_vocab_trace_path:
            self.relation_label_vocab = self.relation_label_vocab[trace]
        self.relation_label_vocab = Vocabulary.load(self.relation_label_vocab)


        tokenizer_trace_path = []
        tokenizer_trace_path_str = self.config['meta_data']['tokenizer']
        if tokenizer_trace_path_str and tokenizer_trace_path_str.strip()!='.':
            tokenizer_trace_path = tokenizer_trace_path_str.split('.')
        assert tokenizer_trace_path, "We need vocab and tokenizer all in meta, so you must provide the trace path from meta"
        tokenizer_config = meta
        for trace in tokenizer_trace_path:
            tokenizer_config = tokenizer_config[trace]
        self.tokenizer = Tokenizer.from_str(tokenizer_config)

        self.save_path = self.config['save_path']
        self.save_root_path = self.config['save_root_path']
        self.start_save_epoch = self.config['start_save_epoch']
        self.start_save_step = self.config['start_save_step']

        self.post_check(self.config, used=[
            "meta",
            "meta_data",
            "relation_types"
            "sym"
            "input_map",
            "origin_input_map",
            "save_root_path",
            "save_path",
            "start_save_step",
            "start_save_epoch",
            "ignore_labels",
        ])


@postprocessor_register('span_relation')
class SpanRelationPostProcessor(IPostProcessor):
    """PostProcess for sequence labeling task"""
    def __init__(self, config: SpanRelationPostProcessorConfig):
        super(SpanRelationPostProcessor, self).__init__(config)
        self.config = config
        self.label_vocab = self.config.label_vocab
        self.tokenizer = self.config.tokenizer
        self.metric = torchmetrics.Accuracy()

    def do_predict(self, stage: str, list_batch_outputs: List[Dict], origin_data: pd.DataFrame, rt_config: Dict)->List:
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
        if self.config.sentence not in origin_data:
            logger.error(f"{self.config.sentence} not in the origin data")
            raise PermissionError(f"{self.config.sentence} must be provided")
        if self.config.uuid not in origin_data:
            logger.error(f"{self.config.uuid} not in the origin data")
            raise PermissionError(f"{self.config.uuid} must be provided")
        predicts = []
        for outputs in list_batch_outputs:
            batch_entity_logits = outputs[self.config.entity_logits]
            batch_relation_logits = outputs[self.config.relation_logits]
            # batch_special_tokens_mask = outputs[self.config.special_tokens_mask]

            indexes = list(outputs[self.config._index])
            for i, (entity_logits, relation_logits, index) in enumerate(zip(batch_entity_logits, batch_relation_logits, indexes)):
                entities_info = self._process4predict(entity_logits, relation_logits, index, origin_data)
        return predicts

    def _process4predict(self, entity_logits: torch.FloatTensor, relation_logits: torch.FloatTensor, index: int, origin_data: pd.DataFrame)->Dict:
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
        one_ins = {}
        origin_ins = origin_data.iloc[int(index)]
        one_ins["sentence"] = origin_ins[self.config.sentence]
        one_ins["uuid"] = origin_ins[self.config.uuid]
        one_ins["entities_info"] = origin_ins[self.config.entities_info]

        word_ids = origin_ins[self.config.word_ids]
        rel_token_len = len(word_ids)
        offset_mapping = origin_ins[self.config.offsets][:rel_token_len]

        predict_entities_info = {}
        predict_entity_ids = entity_logits.argmax(-1).cpu().numpy()

        entity_set_d = {} # set_d, set_e, set_t defined in https://arxiv.org/pdf/2010.13415.pdf Algorithm 1
        for i in range(rel_token_len):
            for j in range(i, rel_token_len):
                if word_ids[i] is None or word_ids[j] is None:
                    continue
                predict_entity_id = predict_entity_ids[i][j]
                predict_entity = self.config.entity_label_vocab[predict_entity_id]
                if predict_entity in {self.config.entity_label_vocab.pad, self.config.entity_label_vocab.unknown}:
                    continue
                else:
                    entity_info = self.get_entity_info([i, j], offset_mapping, word_ids, predict_entity)
                    if entity_info:
                        id = str(uuid.uuid1())
                        predict_entities_info[id] = entity_info
                        entity_set_d[i] = entity_set_d.get(i, [])
                        entity_set_d[i].append((i, j, id))

        entity_tail_pair_set_e = set()
        entity_tail_pair_set_e_with_relation_id = set()
        candidate_relation_set_c = set()
        for relation_idx in self.config.relation_types:
            head_to_head_logits = relation_logits[relation_idx*2]
            tail_to_tail_logits = relation_logits[relation_idx*2+1]
            head_to_head_ids = head_to_head_logits.argmax(-1).cpu().numpy()
            tail_to_tail_ids = tail_to_tail_logits.argmax(-1).cpu().numpy()
            for i in range(rel_token_len):
                for j in range(i if self.config.sym else 0, rel_token_len):
                    if word_ids[i] is None or word_ids[j] is None:
                        continue
                    predict_tail_id = tail_to_tail_ids[i][j]
                    predict_tail_relation = self.config.relation_label_vocab[predict_tail_id]
                    if predict_tail_relation not in {self.config.relation_label_vocab.pad, self.config.relation_label_vocab.unknown}:
                        entity_tail_pair_set_e.add((i, j))
                        entity_tail_pair_set_e_with_relation_id.add((i, j, predict_tail_id))

                    predict_head_id = head_to_head_ids[i][j]
                    predict_head_relation = self.config.relation_label_vocab[predict_head_id]
                    if predict_head_relation not in {self.config.relation_label_vocab.pad, self.config.relation_label_vocab.unknown}:
                        for first_entity in entity_set_d.get(i, []):
                            for second_entity in entity_set_d.get(j, []):
                                candidate_relation_set_c.add((first_entity, second_entity, predict_head_id))
        relations = []
        for candidate_relation in candidate_relation_set_c:
            first_entity, second_entity, predict_head_id = candidate_relation
            if (first_entity[1], second_entity[1]) in entity_tail_pair_set_e:
                # HACK: if (first_entity[1], second_entity[1], predict_head_id) not in entity_tail_pair_set_e_with_relation_id, this is a conflict
                relations.append(candidate_relation)

        # TODO: convert relations to dict
        return relations

    def do_calc_metrics(self, predicts: List, stage: str, list_batch_outputs: List[Dict], origin_data: pd.DataFrame, rt_config: Dict)->Dict:
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

        def _flat_entities_info(entities_info: List[Dict], text: str)->Dict:
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
                label = item['labels'][0]
                if label not in info:
                    info[label] = []
                start_position, end_position = item['start'], item['end']
                while start_position < end_position:
                    if text[start_position] in self.config.ignore_char:
                        start_position += 1
                    else:
                        break
                while start_position < end_position:
                    if text[end_position-1] in self.config.ignore_char:
                        end_position -= 1
                    else:
                        break
                if start_position == end_position: # if the entity after remove ignore char be null, we set it to origin
                    start_position, end_position = item['start'], item['end']
                    
                if self.config.ignore_position:
                    info[label].append(text[item['start']: item['end']].strip())
                else:
                    info[label].append((start_position, end_position))
            return info

        all_predicts = []
        all_ground_truths = []
        for predict in predicts:
            text = predict['sentence']
            predict_ins = _flat_entities_info(predict['predict_entities_info'], text)
            ground_truth_ins = _flat_entities_info(predict['entities_info'], text)
            all_predicts.append(predict_ins)
            all_ground_truths.append(ground_truth_ins)

        precision, recall, f1 = self.calc_score(all_predicts, all_ground_truths)
        real_name = self.loss_name_map(stage)
        logger.info(f'{real_name}_precision: {precision*100}, {real_name}_recall: {recall*100}, {real_name}_f1: {f1*100}')
        return {f'{real_name}_precision': precision*100, f'{real_name}_recall': recall*100, f'{real_name}_f1': f1*100}

    def do_save(self, predicts: List, stage: str, list_batch_outputs: List[Dict], origin_data: pd.DataFrame, rt_config: Dict, save_condition: bool=False):
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
            self.config.start_save_step = rt_config.get('total_steps', 0) - 1
            self.config.start_save_epoch = rt_config.get('total_epochs', 0) - 1
        if rt_config['current_step']>=self.config.start_save_step or rt_config['current_epoch']>=self.config.start_save_epoch:
            save_condition = True
        if save_condition:
            save_path = os.path.join(self.config.save_root_path, self.config.save_path.get(stage, ''))
            if "current_step" in rt_config:
                save_file = os.path.join(save_path, f"step_{str(rt_config['current_step'])}_predict.json")
            else:
                save_file = os.path.join(save_path, 'predict.json')
            logger.info(f"Save the {stage} predict data at {save_file}")
            with open(save_file, 'w') as f:
                json.dump(predicts, f, indent=4, ensure_ascii=False)

    def calc_score(self, predict_list: List, ground_truth_list: List):
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
            """return a/b or 0.0 if b == 0
            """
            if b==0:
                return 0.0
            return a/b


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
            return truth, num_t-truth, num_p-truth

        for predict, ground_truth in zip(predict_list, ground_truth_list):
            keys = set(list(predict.keys())+list(ground_truth.keys()))
            for key in keys:
                tp, fn, fp = _calc_num(predict.get(key, []), ground_truth.get(key, []))
                # logger.info(f"{key} tp num {tp}, fn num {fn}, fp num {fp}")
                category_tp[key] = category_tp.get(key, 0) + tp
                category_fn[key] = category_fn.get(key, 0) + fn
                category_fp[key] = category_fp.get(key, 0) + fp

        all_tp, all_fn, all_fp = 0, 0, 0
        for key in category_tp:
            tp, fn, fp = category_tp[key], category_fn[key], category_fp[key]
            all_tp += tp
            all_fn += fn
            all_fp += fp
            precision = _care_div(tp, tp+fp)
            recall = _care_div(tp, tp+fn)
            f1 = _care_div(2*precision*recall, precision+recall)
            logger.info(f"For entity 「{key}」, the precision={precision*100 :.2f}%, the recall={recall*100:.2f}%, f1={f1*100:.2f}%")
        precision = _care_div(all_tp,all_tp+all_fp)
        recall = _care_div(all_tp, all_tp+all_fn)
        f1 = _care_div(2*precision*recall, precision+recall)
        return precision, recall, f1

    def get_entity_info(self, sub_tokens_index: List, offset_mapping: List, word_ids: List, label: str)->Dict:
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
