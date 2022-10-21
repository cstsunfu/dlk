# Copyright cstsunfu. All rights reserved.
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
from dlk.utils.io import open
import torchmetrics
logger = Logger.get_logger()


@postprocessor_config_register('seq_lab')
class SeqLabPostProcessorConfig(IPostProcessorConfig):
    default_config = {
            "_name": "seq_lab",
            "config": {
                "meta": "*@*",
                "use_crf": False, # use or not use crf
                "word_ready": False, # already gather the subword first token as the word rep or not
                "ignore_position": True, # calc the metrics, whether ignore the ground_truth and predict position info.( if set to true, only focus on the entity content not position.)
                "ignore_char": " ", # if the entity begin or end with this char, will ignore these char
                # "ignore_char": " ()[]-.,:", // if the entity begin or end with this char, will ignore these char
                "meta_data": {
                    "label_vocab": 'label_vocab',
                    "tokenizer": "tokenizer",
                },
                "input_map": {
                    "logits": "logits",
                    "predict_seq_label": "predict_seq_label",
                    "_index": "_index",
                },
                "origin_input_map": {
                    "uuid": "uuid",
                    "sentence": "sentence",
                    "input_ids": "input_ids",
                    "entities_info": "entities_info",
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
                "start_save_step": 0,  # -1 means the last
                "start_save_epoch": -1,
                "aggregation_strategy": "max", # AggregationStrategy item
                "ignore_labels": ['O', 'X', 'S', "E"], # Out, Out, Start, End
            }
        }
    """Config for SeqLabPostProcessor

    Config Example: default_config
    """

    def __init__(self, config: Dict):
        super(SeqLabPostProcessorConfig, self).__init__(config)

        self.use_crf = self.config['use_crf']
        self.word_ready = self.config['word_ready']
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
        self.predict_seq_label = self.input_map['predict_seq_label']
        self._index = self.input_map['_index']

        if isinstance(self.config['meta'], str):
            with open(self.config['meta'], 'rb') as f:
                meta = pkl.load(f)
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
        self.savs_root_path = self.config['save_root_path']
        self.start_save_epoch = self.config['start_save_epoch']
        self.start_save_step = self.config['start_save_step']

        self.post_check(self.config, used=[
            "meta",
            "use_crf",
            "word_ready",
            "meta_data",
            "input_map",
            "origin_input_map",
            "save_root_path",
            "save_path",
            "start_save_step",
            "start_save_epoch",
            "aggregation_strategy",
            "ignore_labels",
            "ignore_char",
            "ignore_position",
        ])


class AggregationStrategy(object):
    """docstring for AggregationStrategy"""
    NONE = "none"
    SIMPLE = "simple"
    FIRST = "first"
    AVERAGE = "average"
    MAX = "max"


@postprocessor_register('seq_lab')
class SeqLabPostProcessor(IPostProcessor):
    """PostProcess for sequence labeling task"""
    def __init__(self, config:    SeqLabPostProcessorConfig):
        super(   SeqLabPostProcessor, self).__init__(config)
        self.config = config
        self.label_vocab = self.config.label_vocab
        self.tokenizer = self.config.tokenizer
        self.metric = torchmetrics.Accuracy()

    def do_predict(self, stage: str, list_batch_outputs: List[Dict], origin_data: pd.DataFrame, rt_config: Dict)->List:
        """Process the model predict to human readable format

        There are three predictor for diffrent seq_lab task dependent on the config.use_crf(the predict is already decoded to ids), and config.word_ready(subword has gathered to firstpiece)

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
        if self.config.use_crf:
            predicts = self.crf_predict(list_batch_outputs=list_batch_outputs, origin_data=origin_data)
        elif self.config.word_ready:
            predicts = self.word_predict(list_batch_outputs=list_batch_outputs, origin_data=origin_data)
        else:
            predicts = self.predict(list_batch_outputs=list_batch_outputs, origin_data=origin_data)
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
        if not save_condition and (rt_config['current_step']>=self.config.start_save_step or rt_config['current_epoch']>=self.config.start_save_epoch):
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
            if key in self.config.ignore_labels:
                continue
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
        if (not sub_tokens_index) or (not label) or (label in self.config.ignore_labels):
            return {}
        start = offset_mapping[sub_tokens_index[0]][0]
        end = offset_mapping[sub_tokens_index[-1]][1]
        return {
            "start": start,
            "end": end,
            "labels": [label]
        }

    def _process4predict(self, predict: torch.LongTensor, index: int, origin_data: pd.DataFrame)->Dict:
        """gather the predict and origin text and ground_truth_entities_info for predict

        Args:
            predict: the predict label_ids
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
        predict = list(predict[:rel_token_len])
        # predict = list(origin_ins['label_ids'][:rel_token_len])
        predict_entities_info = []
        pre_label = ''
        sub_tokens_index = []
        for i, label_id in enumerate(predict):
            if offset_mapping[i] == (0, 0): # added token like [CLS]/<s>/..
                continue
            label = self.config.label_vocab[label_id]
            if label in self.config.ignore_labels \
                or (label[0]=='B') \
                or (label.split('-')[-1] != pre_label):   # label == "O" or label=='B' or label.tail != previor_label
                entity_info = self.get_entity_info(sub_tokens_index, offset_mapping, word_ids, pre_label)
                if entity_info:
                    predict_entities_info.append(entity_info)
                pre_label = ''
                sub_tokens_index = []
            if label not in self.config.ignore_labels:
                assert len(label.split('-')) == 2
                pre_label = label.split('-')[-1]
                sub_tokens_index.append(i)
        entity_info = self.get_entity_info(sub_tokens_index, offset_mapping, word_ids, pre_label)
        if entity_info:
            predict_entities_info.append(entity_info)
        one_ins['predict_entities_info'] = predict_entities_info
        return one_ins

    def crf_predict(self, list_batch_outputs: List[Dict], origin_data: pd.DataFrame)->List:
        """use the crf predict label_ids get predict info

        Args:
            list_batch_outputs: the crf predict info
            origin_data: the origin data

        Returns: 
            all predict instances info

        """
        if self.config.sentence not in origin_data:
            logger.error(f"{self.config.sentence} not in the origin data")
            raise PermissionError(f"{self.config.sentence} must be provided")
        if self.config.uuid not in origin_data:
            logger.error(f"{self.config.uuid} not in the origin data")
            raise PermissionError(f"{self.config.uuid} must be provided")
        if self.config.entities_info not in origin_data:
            logger.error(f"{self.config.entities_info} not in the origin data")
            raise PermissionError(f"{self.config.entities_info} must be provided")

        predicts = []
        for outputs in list_batch_outputs:
            batch_predict = outputs[self.config.predict_seq_label]
            # batch_special_tokens_mask = outputs[self.config.special_tokens_mask]

            indexes = list(outputs[self.config._index])
            outputs = []

            for predict, index in zip(batch_predict, indexes):
                one_ins = self._process4predict(predict, index, origin_data)
                predicts.append(one_ins)
        return predicts

    def word_predict(self, list_batch_outputs: List[Dict], origin_data: pd.DataFrame)->List:
        """use the firstpiece or whole word predict label_logits get predict info

        Args:
            list_batch_outputs: the predict labels logits info
            origin_data: the origin data

        Returns: 
            all predict instances info

        """
        if self.config.sentence not in origin_data:
            logger.error(f"{self.config.sentence} not in the origin data")
            raise PermissionError(f"{self.config.sentence} must be provided")
        if self.config.uuid not in origin_data:
            logger.error(f"{self.config.uuid} not in the origin data")
            raise PermissionError(f"{self.config.uuid} must be provided")
        if self.config.entities_info not in origin_data:
            logger.error(f"{self.config.entities_info} not in the origin data")
            raise PermissionError(f"{self.config.entities_info} must be provided")

        predicts = []
        for outputs in list_batch_outputs:
            batch_logits = outputs[self.config.logits].detach().cpu().numpy()
            # batch_special_tokens_mask = outputs[self.config.special_tokens_mask]

            indexes = list(outputs[self.config._index])

            outputs = []

            for logits, index in zip(batch_logits, indexes):
                origin_ins = origin_data.iloc[int(index)]
                word_ids = origin_ins[self.config.word_ids]

                rel_token_len = len(word_ids)
                logits = logits[:rel_token_len]

                predict = logits.argmax(-1)
                one_ins = self._process4predict(predict, index, origin_data)
                predicts.append(one_ins)
        return predicts

    def predict(self, list_batch_outputs: List[Dict], origin_data: pd.DataFrame)->List:
        """general predict process (especially for subword)

        Args:
            list_batch_outputs: the predict (sub-)labels logits info
            origin_data: the origin data

        Returns: 
            all predict instances info

        """
        if self.config.sentence not in origin_data:
            logger.error(f"{self.config.sentence} not in the origin data")
            raise PermissionError(f"{self.config.sentence} must be provided")
        if self.config.uuid not in origin_data:
            logger.error(f"{self.config.uuid} not in the origin data")
            raise PermissionError(f"{self.config.uuid} must be provided")
        if self.config.entities_info not in origin_data:
            logger.error(f"{self.config.entities_info} not in the origin data")
            raise PermissionError(f"{self.config.entities_info} must be provided")

        predicts = []
        for outputs in list_batch_outputs:
            batch_logits = outputs[self.config.logits].detach().cpu().numpy()
            # batch_special_tokens_mask = outputs[self.config.special_tokens_mask]

            indexes = list(outputs[self.config._index])

            outputs = []

            for logits, index in zip(batch_logits, indexes):
                one_ins = {}
                origin_ins = origin_data.iloc[int(index)]

                input_ids = origin_ins[self.config.input_ids]
                one_ins["sentence"] = origin_ins[self.config.sentence]
                one_ins["uuid"] = origin_ins[self.config.uuid]
                one_ins["entities_info"] = origin_ins[self.config.entities_info]

                rel_token_len = len(input_ids)

                special_tokens_mask = np.array(origin_data.iloc[int(index)][self.config.special_tokens_mask][:rel_token_len])
                offset_mapping = origin_data.iloc[int(index)][self.config.offsets][:rel_token_len]

                logits = logits[:rel_token_len]

                entity_idx = logits.argmax(-1)
                labels = []
                for i, idx in enumerate(list(entity_idx)):
                    labels.append(self.config.label_vocab[idx])

                maxes = np.max(logits, axis=-1, keepdims=True)
                shifted_exp = np.exp(logits - maxes)
                scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

                pre_entities = self.gather_pre_entities(
                    one_ins["sentence"], input_ids, scores, offset_mapping, special_tokens_mask)
                grouped_entities = self.aggregate(pre_entities, self.config.aggregation_strategy)
                # Filter anything that is in self.ignore_labels
                entities = [
                    entity
                    for entity in grouped_entities
                    if entity.get("entity", None) not in self.config.ignore_labels
                    and entity.get("entity_group", None) not in self.config.ignore_labels
                ]
                predict_entities_info = []
                for entity in entities:
                    one_predict = {}
                    one_predict['start'] = entity['start']
                    one_predict['end'] = entity['end']
                    one_predict['labels'] = [entity['entity_group']]
                    predict_entities_info.append(one_predict)
                one_ins['predict_entities_info'] = predict_entities_info
                predicts.append(one_ins)
        return predicts

    def aggregate(self, pre_entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        if aggregation_strategy in {AggregationStrategy.NONE, AggregationStrategy.SIMPLE}:
            entities = []
            for pre_entity in pre_entities:
                entity_idx = pre_entity["scores"].argmax()
                score = pre_entity["scores"][entity_idx]
                entity = {
                    "entity": self.config.label_vocab[entity_idx],
                    "score": score,
                    "index": pre_entity["index"],
                    "word": pre_entity["word"],
                    "start": pre_entity["start"],
                    "end": pre_entity["end"],
                }
                entities.append(entity)
        else:
            entities = self.aggregate_words(pre_entities, aggregation_strategy)

        if aggregation_strategy == AggregationStrategy.NONE:
            return entities
        return self.group_entities(entities)

    def aggregate_word(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> dict:
        word = self.tokenizer.decode([self.tokenizer.token_to_id(entity['word']) for entity in entities])
        if aggregation_strategy == AggregationStrategy.FIRST:
            scores = entities[0]["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.config.label_vocab[idx]
        elif aggregation_strategy == AggregationStrategy.MAX:
            max_entity = max(entities, key=lambda entity: entity["scores"].max())
            scores = max_entity["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.config.label_vocab[idx]
        elif aggregation_strategy == AggregationStrategy.AVERAGE:
            scores = np.stack([entity["scores"] for entity in entities])
            average_scores = np.nanmean(scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.config.label_vocab[entity_idx]
            score = average_scores[entity_idx]
        else:
            raise ValueError("Invalid aggregation_strategy")
        new_entity = {
            "entity": entity,
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return new_entity


    def group_sub_entities(self, entities: List[dict]) -> dict:
        """Group together the adjacent tokens with the same entity predicted.

        Args:
            entities: The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]
        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": " ".join(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def get_tag(self, entity_name: str) -> Tuple[str, str]:
        if entity_name.startswith("B-"):
            bi = "B"
            tag = entity_name[2:]
        elif entity_name.startswith("I-"):
            bi = "I"
            tag = entity_name[2:]
        else:
            # It's not in B-, I- format
            # Default to I- for continuation.
            bi = "I"
            tag = entity_name
        return bi, tag

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities: The entities predicted by the pipeline.
        """

        entity_groups = []
        entity_group_disagg = []

        for entity in entities:
            if not entity_group_disagg:
                entity_group_disagg.append(entity)
                continue

            # If the current entity is similar and adjacent to the previous entity,
            # append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" prefixes
            # Shouldn't merge if both entities are B-type
            bi, tag = self.get_tag(entity["entity"])
            last_bi, last_tag = self.get_tag(entity_group_disagg[-1]["entity"])

            if tag == last_tag and bi != "B":
                # Modify subword type to be previous_type
                entity_group_disagg.append(entity)
            else:
                # If the current entity is different from the previous entity
                # aggregate the disaggregated entity group
                entity_groups.append(self.group_sub_entities(entity_group_disagg))
                entity_group_disagg = [entity]
        if entity_group_disagg:
            # it's the last entity, add it to the entity groups
            entity_groups.append(self.group_sub_entities(entity_group_disagg))

        return entity_groups

    def aggregate_words(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        """Override tokens from a given word that disagree to force agreement on word boundaries.

        Example: 
            micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft|
            company| B-ENT I-ENT
        """
        if aggregation_strategy in {
            AggregationStrategy.NONE,
            AggregationStrategy.SIMPLE,
        }:
            raise ValueError("NONE and SIMPLE strategies are invalid for word aggregation")

        word_entities = []
        word_group = None
        for entity in entities:
            if word_group is None:
                word_group = [entity]
            elif entity["is_subword"]:
                word_group.append(entity)
            else:
                word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
                word_group = [entity]
        # Last item
        word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
        return word_entities

    def gather_pre_entities(
        self,
        sentence: str,
        input_ids: np.ndarray,
        scores: np.ndarray,
        offset_mapping: Optional[List[Tuple[int, int]]],
        special_tokens_mask: np.ndarray,
    ) -> List[dict]:
        """Fuse various numpy arrays into dicts with all the information needed for aggregation"""
        pre_entities = []
        for idx, token_scores in enumerate(scores):
            # Filter special_tokens, they should only occur
            # at the sentence boundaries since we're not encoding pairs of
            # sentences so we don't have to keep track of those.
            if special_tokens_mask[idx]:
                continue

            word = self.tokenizer.id_to_token(int(input_ids[idx]))
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                word_ref = sentence[start_ind:end_ind]
                if getattr(self.tokenizer.model, "continuing_subword_prefix", None):
                    # This is a BPE, word aware tokenizer, there is a correct way
                    # to fuse tokens
                    is_subword = len(word) != len(word_ref)
                else:
                    # This is a fallback heuristic. This will fail most likely on any kind of text + punctuation mixtures that will be considered "words". Non word aware models cannot do better than this unfortunately.
                    is_subword = sentence[start_ind - 1 : start_ind] != " " if start_ind > 0 else False

            else:
                start_ind = None
                end_ind = None
                is_subword = False

            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            pre_entities.append(pre_entity)
        return pre_entities

