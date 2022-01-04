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

from dlk.utils.vocab import Vocabulary
from dlk.utils.config import BaseConfig, ConfigTool
from typing import Dict, Callable, Iterable, Set, List
from dlk.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from dlk.utils.logger import Logger
import pandas as pd

logger = Logger.get_logger()

@subprocessor_config_register('token_gather')
class TokenGatherConfig(BaseConfig):
    """Config for TokenGather

    Config Example:
        >>> {
        >>>     "_name": "token_gather",
        >>>     "config": {
        >>>         "train": { // only train stage using
        >>>             "data_set": {                   // for different stage, this processor will process different part of data
        >>>                 "train": ["train", "valid", 'test']
        >>>             },
        >>>             "gather_columns": "*@*", //List of columns, if one element of the list is dict, you can set more. Every cell must be sigle token or list of tokens or set of tokens
        >>>             //"gather_columns": ['tokens']
        >>>             //"gather_columns": ['tokens', {"column": "entities_info", "trace": 'labels'}] 
        >>>             // the trace only trace the dict, if list is in trace path, will add the trace to every elements in the list. for example: {"entities_info": [{'start': 1， 'end': 2, labels: ['Label1']}, ..]}, the trace to labels is 'entities_info.labels'
        >>>             "deliver": "*@*", // output Vocabulary object (the Vocabulary of labels) name.
        >>>             "ignore": "", // ignore the token, the id of this token will be -1
        >>>             "update": null, // null or another Vocabulary object to update
        >>>             "unk": "[UNK]",
        >>>             "pad": "[PAD]",
        >>>             "min_freq": 1,
        >>>             "most_common": -1, //-1 for all
        >>>         }
        >>>     }
        >>> }
    """

    def __init__(self, stage: str, config: Dict):
        super(TokenGatherConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])
        if not self.data_set:
            return
        self.ignore = self.config['ignore']
        self.gather_columns = self.config["gather_columns"]
        self.deliver = self.config["deliver"]
        if self.data_set and (not self.deliver):
            raise ValueError("The 'deliver' value must not be null.")
        self.update = self.config['update']
        self.unk = self.config['unk']
        self.pad = self.config['pad']
        self.min_freq = self.config['min_freq']
        self.most_common = self.config['most_common']
        self.post_check(self.config, used=[
            "data_set",
            "gather_columns",
            "deliver",
            "ignore",
            "update",
            "unk",
            "pad",
            "min_freq",
            "most_common",
        ])

@subprocessor_register('token_gather')
class TokenGather(ISubProcessor):
    """gather all tokens from the 'gather_columns' and deliver a vocab named 'token_vocab'
    """
    def __init__(self, stage: str, config: TokenGatherConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.data_set = config.data_set
        if not self.data_set:
            logger.info(f"Skip 'token_gather' at stage {self.stage}")
            return
        self.update = config.update

    def get_elements_from_series_by_trace(self, data: pd.Series, trace: str)->List:
        """get the datas from data[trace_path]
        >>> for example:
        >>> data[0] = {'entities_info': [{'start': 0, 'end': 1, 'labels': ['Label1']}]} // data is a series, and every element is as data[0]
        >>> trace = 'entities_info.labels'
        >>> return_result = [['Label1']]

        Args:
            data: origin data series
            trace: get data element trace

        Returns: 
            the data in the tail of traces

        """

        def get_elements_from_iter_by_trace(iter: Iterable, cur_trace_list: List):
            if not cur_trace_list:
                return iter
            if isinstance(iter, dict):
                return get_elements_from_iter_by_trace(iter[cur_trace_list[0]], cur_trace_list[1:])
            if isinstance(iter, list) or  isinstance(iter, tuple):
                return [get_elements_from_iter_by_trace(sub_iter, cur_trace_list) for sub_iter in iter]
            raise PermissionError(f"The trace path is only support type list and dict, but you provide {type(iter)}")
            
        return [get_elements_from_iter_by_trace(one, trace.split('.')) for one in data]

    def process(self, data: Dict)->Dict:
        """TokenGather entry

        Args:
            data: 
            >>> {
            >>>     "data": {"train": ...},
            >>>     "tokenizer": ..
            >>> }

        Returns: 
            data[self.config.deliver] = Vocabulary()(which gathered_token)

        """
        if not self.data_set:
            return data
        if self.update:
            self.vocab = data[self.update]
        else:
            self.vocab = Vocabulary(do_strip=True, unknown=self.config.unk, ignore=self.config.ignore, pad=self.config.pad)
        for data_set_name in self.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip gather tokens from it.')
                continue
            data_set = data['data'][data_set_name]
            for column in self.config.gather_columns:
                if isinstance(column, str):
                    self.vocab.auto_update(data_set[column])
                elif isinstance(column, dict):
                    self.vocab.auto_update(self.get_elements_from_series_by_trace(data_set[column['column']], trace=column['trace']))
                else:
                    raise PermissionError(f'The gather column currently is only support str or dict.')
        self.vocab.filter_rare(self.config.min_freq, self.config.most_common)
        logger.info(f"The Vocab Num is {self.vocab.word_num}")
        data[self.config.deliver] = self.vocab.__dict__
        return data
