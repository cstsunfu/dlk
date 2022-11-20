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
import pytorch_lightning as pl
from dlk.process import Processor
from dlk.predict import Predict
import json
import torch

logger = Logger('log.txt')

pl.seed_everything(88)

with open('./test.json', 'r') as f:
    data = json.load(f)

# train = data[:len(data)//2]
valid = data[:3]
input = {"data": {"train": valid}}

processor = Processor('./bert/prepro.hjson')
data = processor.fit(input, stage='train')
# print(data.keys())
print(data['data']['train'])
predict = Predict('./bert/main.hjson', "./bert/output/task.bert/lightning_logs/checkpoints/epoch=0-step=1.ckpt")

# model = torch.load('./bert/output/task.bert/lightning_logs/checkpoints/epoch=0-step=1.ckpt', map_location="cpu")
# for key in model['state_dict']:
#     print(key)


predict.convert2script(data['data'])

