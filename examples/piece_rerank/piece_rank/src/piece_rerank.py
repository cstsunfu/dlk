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
import copy
from typing import Dict
from dlk.data.postprocessors import postprocessor_register, postprocessor_config_register, IPostProcessor, IPostProcessorConfig
from dlk.utils.logger import Logger
from dlk.utils.vocab import Vocabulary
from dlk.utils.io import open
from tokenizers import Tokenizer
logger = Logger.get_logger()


class BeamNode():
    """docstring for Beam"""
    def __init__(self, logits):
        super(BeamNode, self).__init__()
        self.visited = []
        self.score = 0.0
        self.logits = logits

    def update(self, score, position):
        """TODO: Docstring for update.
        Returns: TODO

        """
        self.score += score
        self.visited.append(position)
        self.logits[:, position] = -np.inf

class Beam(object):
    """docstring for Beam"""
    def __init__(self, beam_size, start_index, end_index, predict_logits):
        super(Beam, self).__init__()
        self.beam_nodes: List[BeamNode] = []
        self.beam_size = beam_size
        self.start_index = start_index
        self.end_index = end_index

        values, indices = torch.topk(predict_logits[start_index][start_index+1: end_index], beam_size)
        for value, position in zip(values, indices):
            beam_node = BeamNode(copy.deepcopy(predict_logits))
            beam_node.update(value.item(), position.item()+start_index+1)
            self.beam_nodes.append(beam_node)

        for _ in range(end_index - start_index - 2):
            self.step()
            self.beam_nodes.sort(key=lambda node: node.score, reverse=True)
            self.beam_nodes = self.beam_nodes[:beam_size]

        for beam_node in self.beam_nodes:
            beam_node.score = predict_logits[beam_node.visited[-1]][start_index]
        self.beam_nodes.sort(key=lambda node: node.score, reverse=True)

    def step(self):
        beam_nodes = []
        for beam_node in self.beam_nodes:
            values, indices = torch.topk(beam_node.logits[beam_node.visited[-1]][self.start_index+1: self.end_index], self.beam_size)
            for i, (value, position) in enumerate(zip(values, indices)):
                if i == self.beam_size - 1:
                    new_beam_node = beam_node # reuse
                else:
                    new_beam_node = copy.deepcopy(beam_node)
                new_beam_node.update(value.item(), position.item()+self.start_index+1)
                beam_nodes.append(new_beam_node)
        self.beam_nodes = beam_nodes


@postprocessor_config_register('piece_rerank')
class PieceRerankPostProcessorConfig(IPostProcessorConfig):
    default_config = {
        "_name": "piece_rerank",
        "config": {
            "meta": "*@*",
            "ignore_char": " ", # if the answer begin or end with this char, will ignore these char
            # "ignore_char": " ()[]-.,:", # if the answer begin or end with this char, will ignore these char
            "meta_data": {
                "tokenizer": "tokenizer",
            },
            "input_map": {
                "logits": "logits",
                "_index": "_index",
            },
            "origin_input_map": {
                "uuid": "uuid",
                "pretokenized_words": "pretokenized_words", # for pair inputs, tokenize the "sentence_a" && "sentence_b"
                "input_ids": "input_ids",
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
            "ignore_labels": ['O', '[UNK]', '[PAD]']
        }
    }
    """Config for PieceRerankPostProcessor

    Config Example:
    """

    def __init__(self, config: Dict):
        super(PieceRerankPostProcessorConfig, self).__init__(config)
        self.ignore_labels = set(self.config['ignore_labels'])
        self.ignore_char = set(self.config['ignore_char'])

        self.pretokenized_words = self.origin_input_map['pretokenized_words']
        self.offsets = self.origin_input_map['offsets']
        self.uuid = self.origin_input_map['uuid']
        self.word_ids = self.origin_input_map['word_ids']
        self.special_tokens_mask = self.origin_input_map['special_tokens_mask']
        self.input_ids = self.origin_input_map['input_ids']
        self.label_ids = self.origin_input_map['label_ids']

        self.logits = self.input_map['logits']
        self._index = self.input_map['_index']

        if isinstance(self.config['meta'], str):
            with open(self.config['meta'], 'rb') as f:
                meta = pkl.load(f)
        else:
            raise PermissionError("You must provide meta data(vocab & tokenizer) for ner postprocess.")


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
            "ignore_labels",
        ])


@postprocessor_register('piece_rerank')
class PieceRerankPostProcessor(IPostProcessor):
    """PostProcess for sequence labeling task"""
    def __init__(self, config: PieceRerankPostProcessorConfig):
        super(PieceRerankPostProcessor, self).__init__(config)
        self.tokenizer = self.config.tokenizer

    def _process4predict(self, predict_logits: torch.Tensor, index: int, origin_data: pd.DataFrame)->Dict:
        """gather the predict and origin text and ground_truth_entities_info for predict

        Args:
            predict: the predict span logits
            index: the data index in origin_data
            origin_data: the origin pd.DataFrame

        Returns: 
            >>> one_ins info 
            >>> {
            >>>     "uuid": "..",
            >>> }

        """

        one_ins = {}
        origin_ins = origin_data.iloc[int(index)]
        one_ins["uuid"] = origin_ins[self.config.uuid]

        word_ids = origin_ins[self.config.word_ids]
        rel_token_len = len(word_ids)
        start_index = word_ids.index(0)
        end_index = rel_token_len
        while word_ids[end_index-1] is None:
            end_index -= 1
            
        beams = Beam(beam_size=3, start_index=start_index, end_index=end_index, predict_logits=predict_logits)
        # print(beams.beam_nodes[0].score)

        visited = [0]
        visited_set = set()
        for visit in beams.beam_nodes[0].visited:
            cur_visit_word = word_ids[visit]
            if cur_visit_word not in visited_set:
                visited_set.add(cur_visit_word)
                visited.append(cur_visit_word)
        print(f"index {int(index)}:", " ".join(origin_ins['pretokenized_words'][visit] for visit in visited[1:]), "--->", " ".join(origin_ins['pretokenized_words'][visit] for visit in origin_ins['rank_info'][1:]))

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
        if self.config.uuid not in origin_data:
            logger.error(f"{self.config.uuid} not in the origin data")
            raise PermissionError(f"{self.config.uuid} must be provided")

        predicts = []
        for outputs in list_batch_outputs:
            batch_logits = torch.softmax(outputs[self.config.logits], dim=-1)[:, :, :, 1]
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
        return {"score": 0}

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
        pass
        # if self.config.start_save_epoch == -1 or self.config.start_save_step == -1:
        #     self.config.start_save_step = rt_config.get('total_steps', 0) - 1
        #     self.config.start_save_epoch = rt_config.get('total_epochs', 0) - 1
        # if rt_config['current_step']>=self.config.start_save_step or rt_config['current_epoch']>=self.config.start_save_epoch:
        #     save_condition = True
        # if save_condition:
        #     save_path = os.path.join(self.config.save_root_path, self.config.save_path.get(stage, ''))
        #     if "current_step" in rt_config:
        #         save_file = os.path.join(save_path, f"step_{str(rt_config['current_step'])}_predict.json")
        #     else:
        #         save_file = os.path.join(save_path, 'predict.json')
        #     logger.info(f"Save the {stage} predict data at {save_file}")
        #     with open(save_file, 'w') as f:
        #         json.dump(predicts, f, indent=4, ensure_ascii=False)
