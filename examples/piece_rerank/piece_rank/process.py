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

from datasets import load_dataset
from dlk.utils.logger import Logger
import copy
import random
import json
import hjson
from dlk.process import Processor
import uuid
import numpy as np
import os
# os.envDISABLE_PANDAS_PARALLEL
# os.environ["DISABLE_PANDAS_PARALLEL"] = "true"
random.seed(1)
logger = Logger('log.txt')
# {
#     "uuid": "uuid",
#     "pretokenized_words": ["B", "A", "C"],
#     "rank_info": [1, 0, 2]
# }

data = []
with open('./english.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        tokens = line.split()
        tokens_index = [(i, c) for i, c in enumerate(tokens)]
        random.shuffle(tokens_index)
        rank_info = np.argsort([i for (i,_) in tokens_index]) + 1
        pretokenized_words = ['[PAD]'] + [c for (_, c) in tokens_index]
        data.append({
            "uuid": str(uuid.uuid4()),
            "pretokenized_words": pretokenized_words,
            "rank_info": [0] + list(rank_info)
        })
# data = [
#     {
#         "uuid": "uuid",
#         "pretokenized_words": ['[PAD]', "BCD", "E", "A"],
#         "rank_info": [0, 3, 1, 2]
#     }
# ] * 10


input = {"data": {"train": data[:106], "valid": data[101:]}}
processor = Processor('./bert/prepro.hjson')
# print(json.dumps(processor.config, indent=4))
processor.fit(input)
