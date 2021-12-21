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
from dlk.predict import Predict
import json

Logger("./conll_norm_char_lstm_crf.log")
pl.seed_everything(88)

predictor = Predict('./examples/sequence_labeling/conll2003/norm_char_lstm_crf/output/task=crf_lstm_lr=0.01_optimizer=sgd_dropout=0.5_batch_size=10_lstm_output_size=200/config.json', './examples/sequence_labeling/conll2003/norm_char_lstm_crf/output/task=crf_lstm_lr=0.01_optimizer=sgd_dropout=0.5_batch_size=10_lstm_output_size=200/default/checkpoints/epoch=0-step=1404.ckpt')

predictor.predict()
