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

from dlk.data.datamodules import datamodule_register, datamodule_config_register
from dlk.data.postprocessors import postprocessor_register, postprocessor_config_register
from dlk.data.processors import processor_config_register, processor_register
from dlk.data.subprocessors import subprocessor_config_register, subprocessor_register
