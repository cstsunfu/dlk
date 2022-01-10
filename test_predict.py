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
from dlk.predict import Predict
from dlk.utils.parser import BaseConfigParser
from dlk.data.processors import processor_register
import pandas as pd
import pytorch_lightning as pl
import json
import copy
import hjson

Logger("./test_predict.log")


predict_data = {} # this should fill the predict_data
data = {"predict": predict_data}

inp = {"data": data}

prepro_config = hjson.load(open("path/to/prepro.hjson"), object_pairs_hook=dict)

prepro_config['config']['data_dir'] = 'path/to/meta.pkl'
# ...
# other will update
# ...
prepro_config['config']['feed_order'].remove('save')

prepro_config = BaseConfigParser(prepro_config).parser_with_check()
assert len(prepro_config) == 1
prepro_config = prepro_config[0]['processor']

# print(json.dumps(config, indent=4))

processed_data = processor_register.get(prepro_config.get('_name'))(stage="predict", config=prepro_config).process(inp)

main_config = hjson.load(open("path/to/main.hjson"), object_pairs_hook=dict)
main_config['config']['meta_data'] = 'pata/to/meta.pkl'
# ...
# other will update
# ...

predictor = Predict(main_config, 'path/to/checkpoint_path.hjson')

predict_data = predictor.predict(data=processed_data, save_condition=False)
