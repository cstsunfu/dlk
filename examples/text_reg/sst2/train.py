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
from dlk.train import Train
import json

logger = Logger('log.txt')
pl.seed_everything(88)

# trainer = Train('./examples/sequence_labeling/benchmark/pretrained/first_piece_lstm_crf_main.hjson')
trainer = Train('./distil_bert/main.hjson')
# print(json.dumps(trainer.configs, indent=4))

trainer.run()
