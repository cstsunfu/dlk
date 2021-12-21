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
from dlk.utils.logger import Logger
from dlk.utils.parser import config_parser_register
import copy
import json
import hjson
from dlk.data.processors import processor_config_register, processor_register

Logger('process.log')
train_data = json.load(open('./examples/sequence_labeling/conll2003/data/train.json', 'r'))
valid_data = json.load(open('./examples/sequence_labeling/conll2003/data/valid.json', 'r'))
test_data = json.load(open('./examples/sequence_labeling/conll2003/data/test.json', 'r'))
data = {"predict": test_data}

inp = {"data": data}
config = config_parser_register.get("processor")(hjson.load(open("./examples/sequence_labeling/conll2003/norm_char_lstm_crf/prepro.hjson"),object_pairs_hook=dict)).parser_with_check()[0]
config['data_dir'] = 'test_predict'

# print(json.dumps(config, indent=4))
processor_register.get(config.get('_name'))(stage="predict", config=processor_config_register.get(config.get('_name'))(stage="train", config=config)).process(inp)
