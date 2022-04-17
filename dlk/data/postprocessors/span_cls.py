# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from tokenizers import Tokenizer
import torchmetrics
logger = Logger.get_logger()


@postprocessor_config_register('span_cls')
class SpanClsPostProcessorConfig(IPostProcessorConfig):
    """Config for SpanClsPostProcessor

    Config Example:
        >>> {
        >>>     "_name": "span_cls",
        >>>     "config": {
        >>>         "meta": "*@*",
        >>>         "ignore_position": false, // calc the metrics, whether ignore the ground_truth and predict position info.( if set to true, only focus on the entity content not position.)
        >>>         "ignore_char": " ", // if the entity begin or end with this char, will ignore these char
        >>>         //"ignore_char": " ()[]-.,:", // if the entity begin or end with this char, will ignore these char
        >>>         "meta_data": {
        >>>             "label_vocab": 'label_vocab',
        >>>             "tokenizer": "tokenizer",
        >>>         },
        >>>         "input_map": {
        >>>             "logits": "logits",
        >>>             "_index": "_index",
        >>>         },
        >>>         "origin_input_map": {
        >>>             "uuid": "uuid",
        >>>             "sentence": "sentence",
        >>>             "input_ids": "input_ids",
        >>>             "entities_info": "entities_info",
        >>>             "offsets": "offsets",
        >>>             "special_tokens_mask": "special_tokens_mask",
        >>>             "word_ids": "word_ids",
        >>>             "label_ids": "label_ids",
        >>>         },
        >>>         "save_root_path": ".",  //save data root dir
        >>>         "save_path": {
        >>>             "valid": "valid",  // relative dir for valid stage
        >>>             "test": "test",    // relative dir for test stage
        >>>         },
        >>>         "start_save_step": 0,  // -1 means the last
        >>>         "start_save_epoch": -1,
        >>>         "aggregation_strategy": "max", // AggregationStrategy item
        >>>         "ignore_labels": ['O', 'X', 'S', "E"], // Out, Out, Start, End
        >>>     }
        >>> }
    """

    def __init__(self, config: Dict):
        super(SpanClsPostProcessorConfig, self).__init__(config)

        self.aggregation_strategy = self.config['aggregation_strategy']
        self.ignore_labels = set(self.config['ignore_labels'])
        self.ignore_char = set(self.config['ignore_char'])
        self.ignore_position = self.config['ignore_position']

        self.sentence = self.origin_input_map['sentence']
        self.offsets = self.origin_input_map['offsets']
        self.entities_info = self.origin_input_map['entities_info']
        self.uuid = self.origin_input_map['uuid']
        self.word_ids = self.origin_input_map['word_ids']
        self.special_tokens_mask = self.origin_input_map['special_tokens_mask']
        self.input_ids = self.origin_input_map['input_ids']
        self.label_ids = self.origin_input_map['label_ids']

        self.logits = self.input_map['logits']
        self._index = self.input_map['_index']

        if isinstance(self.config['meta'], str):
            meta = pkl.load(open(self.config['meta'], 'rb'))
        else:
            raise PermissionError("You must provide meta data(vocab & tokenizer) for ner postprocess.")

        vocab_trace_path = []
        vocab_trace_path_str = self.config['meta_data']['label_vocab']
        if vocab_trace_path_str and vocab_trace_path_str.strip()!='.':
            vocab_trace_path = vocab_trace_path_str.split('.')
        assert vocab_trace_path, "We need vocab and tokenizer all in meta, so you must provide the trace path from meta"
        self.label_vocab = meta
        for trace in vocab_trace_path:
            self.label_vocab = self.label_vocab[trace]
        self.label_vocab = Vocabulary.load(self.label_vocab)


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
            "input_map",
            "origin_input_map",
            "save_root_path",
            "save_path",
            "start_save_step",
            "start_save_epoch",
            "aggregation_strategy",
            "ignore_labels",
        ])


@postprocessor_register('span_cls')
class SpanClsPostProcessor(IPostProcessor):
    """PostProcess for sequence labeling task"""
    def __init__(self, config:    SpanClsPostProcessorConfig):
        super(   SpanClsPostProcessor, self).__init__()
        self.config = config
        self.label_vocab = self.config.label_vocab
        self.tokenizer = self.config.tokenizer
        self.metric = torchmetrics.Accuracy()

    def _process4predict(self, predict_logits: torch.FloatTensor, index: int, origin_data: pd.DataFrame)->Dict:
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
        one_ins = {}
        origin_ins = origin_data.iloc[int(index)]
        one_ins["sentence"] = origin_ins[self.config.sentence]
        one_ins["uuid"] = origin_ins[self.config.uuid]
        one_ins["entities_info"] = origin_ins[self.config.entities_info]

        word_ids = origin_ins[self.config.word_ids]
        rel_token_len = len(word_ids)
        offset_mapping = origin_ins[self.config.offsets][:rel_token_len]

        predict_entities_info = []
        predict_label_ids = predict_logits.argmax(-1).cpu().numpy()
        for i in range(rel_token_len):
            for j in range(i, rel_token_len):
                predict_label_id = predict_label_ids[i][j]
                predict_label = self.config.label_vocab[predict_label_id]
                if predict_label == self.config.label_vocab.pad or predict_label == self.config.label_vocab.unknown:
                    continue
                else:
                    entity_info = self.get_entity_info([i, j], offset_mapping, word_ids, predict_label)
                    if entity_info:
                        predict_entities_info.append(entity_info)
        one_ins['predict_entities_info'] = predict_entities_info
        return one_ins

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
            batch_logits = outputs[self.config.logits]
            # batch_special_tokens_mask = outputs[self.config.special_tokens_mask]

            indexes = list(outputs[self.config._index])

            for i, (predict, index) in enumerate(zip(batch_logits, indexes)):
                one_ins = self._process4predict(predict, index, origin_data)
                one_ins['predict_extend_return'] = self.gather_predict_extend_data(outputs, i, self.config.predict_extend_return)
                predicts.append(one_ins)
        return predicts

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
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            if "current_step" in rt_config:
                save_file = os.path.join(save_path, f"step_{str(rt_config['current_step'])}_predict.json")
            else:
                save_file = os.path.join(save_path, 'predict.json')
            logger.info(f"Save the {stage} predict data at {save_file}")
            json.dump(predicts, open(save_file, 'w'), indent=4, ensure_ascii=False)

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
            "labels": [label]
        }
