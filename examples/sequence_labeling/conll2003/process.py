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

from datasets import load_dataset
from dlk.utils.logger import Logger
import copy
import json
import hjson
from dlk.process import Processor
from utils import convert
import uuid

logger = Logger('log.txt')


# this is just for prepro the data, not the real label<->id pair in process.
label_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}

data = load_dataset(path="conll2003")

data = data.map(lambda one: {"tokens": one["tokens"], 'ner_tags': [label_map[i] for i in one['ner_tags']]})


filed_name_map = {"train": "train", "validation": "valid", 'test': "test"}
json_data_map = {}
for filed in ['train', 'validation', 'test']:
    filed_data = data[filed].to_dict()
    tokens = filed_data['tokens']
    labels = filed_data['ner_tags']
    inses = []
    for token, label in zip(tokens, labels):
        inses.append([token, label])
    json_data_map[filed_name_map[filed]] = convert(inses)


input = {"data": json_data_map}

processor = Processor('./norm_char_lstm_crf/prepro.hjson')
processor.fit(input)
