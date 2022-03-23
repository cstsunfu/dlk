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
from dlk.core.imodels import imodel_config_register, imodel_register
from dlk.core.models import model_config_register, model_register
from dlk.core.callbacks import callback_config_register, callback_register
from dlk.core.initmethods import initmethod_config_register, initmethod_register
from dlk.core.layers import embedding_register, embedding_config_register, decoder_config_register, decoder_register, encoder_config_register, encoder_register
from dlk.core.optimizers import optimizer_config_register, optimizer_register
from dlk.core.schedulers import scheduler_config_register, scheduler_register
from dlk.core.losses import loss_config_register, loss_register
from dlk.core.modules import module_config_register, module_register
