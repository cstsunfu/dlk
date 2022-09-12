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

from dlk.utils.logger import Logger
import copy
import json
import hjson
from dlk.process import Processor
import uuid

logger = Logger('log.txt')

with open('./data/train.tags.de-en.clean.de') as f:
    de_lines = f.readlines()

with open('./data/train.tags.de-en.clean.en') as f:
    en_lines = f.readlines()

with open('./data/target.txt') as f:
    de_lines = f.readlines()

with open('./data/source.txt') as f:
    en_lines = f.readlines()

data = []
for en_line, de_line in zip(en_lines, de_lines):
    data.append({
        # "encoder": f"[MASK] {en_line} [MASK]",
        # "decoder": f"[MASK] {de_line} [MASK]",
        "encoder": f"{en_line}",
        "decoder": f"{de_line}",
        "uuid": str(uuid.uuid1()),
    })
input = {"data": {"train": data[:1000], 'valid': data[:10]}}

processor = Processor('./transformer/prepro.hjson')
# print(json.dumps(processor.config, indent=2))
processor.fit(input)
