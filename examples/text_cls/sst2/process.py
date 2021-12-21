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
import uuid

logger = Logger('log.txt')

def flat(data):
    """TODO: Docstring for flat.

    :data: TODO
    :returns: TODO

    """
    sentences = data['sentence']
    uuids = data['uuid']
    labelses = data['labels']
    return [{'sentence': sentece, 'labels': [labels], "uuid": uuid} for sentece, labels, uuid in zip(sentences, labelses, uuids)]

label_map = {0: "neg", 1: 'pos'}

data = load_dataset("gpt3mix/sst2")
data = data.map(lambda one: {"sentence": one["text"], 'labels': label_map[one['label']], "uuid": str(uuid.uuid1())}, remove_columns=['text', 'label'])
input = {"data": {"train": flat(data['train'].to_dict()), 'valid': flat(data['validation'].to_dict()), 'test': flat(data['test'].to_dict())}}

processor = Processor('./distil_bert/prepro.hjson')
processor.fit(input)
