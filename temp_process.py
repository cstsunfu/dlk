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

import pandas as pd
from dlk.utils.parser import config_parser_register
import copy
import json
import hjson
from dlk.process import Processor

train_data = json.load(open('./examples/sequence_labeling/conll2003/data/train.json', 'r'))
valid_data = json.load(open('./examples/sequence_labeling/conll2003/data/valid.json', 'r'))
test_data = json.load(open('./examples/sequence_labeling/conll2003/data/test.json', 'r'))
data = {"train": train_data, "valid": valid_data, "test": test_data}

inp = {"data": data}
processor = Processor('./examples/sequence_labeling/conll2003/norm_char_lstm_crf/prepro.hjson')
processor.fit({})
