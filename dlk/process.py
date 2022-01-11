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

from dlk.utils.parser import BaseConfigParser
import json
import hjson
from typing import Dict, Union, Any
from dlk.data.processors import processor_config_register, processor_register


class Processor(object):
    """Processor"""
    def __init__(self, config: Union[str, Dict]):
        super(Processor, self).__init__()
        if not isinstance(config, dict):
            config = hjson.load(open(config), object_pairs_hook=dict)
            config = BaseConfigParser(config).parser_with_check()
            assert len(config) == 1, f"Currently we didn't support search for Processor, if you require this feature please create an issue to describe the reason details."
            self.config = config[0]
        self.config = self.config['processor']
        
    def fit(self, data: Dict[str, Any], stage='train'):
        """Process the data and return the processed data

        Args:
            data: {"train": .., 'valid': ..}
            stage: "train"/ 'predict', etc.

        Returns: 
            processed data

        """
        
        processor = processor_register.get(self.config.get('_name'))(stage=stage, config=processor_config_register.get(self.config.get('_name'))(stage=stage, config=self.config))
        return processor.process(data)
