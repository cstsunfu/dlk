# Copyright cstsunfu. All rights reserved.
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

from dlk.predict import Predict
from dlk.utils.logger import Logger
from typing import Dict, Union, Callable, List, Any
import torch

logger = Logger.get_logger()


class OnlinePredict(Predict):
    """OnlinePredict
    """
    def __init__(self, config: Union[str, dict], checkpoint: str):
        super(OnlinePredict, self).__init__(config, checkpoint)
        self.datamodule = self.get_datamodule(self.config, {})

    def predict(self, data):
        """init the model, datamodule, manager then predict the predict_dataloader

        Args:
            data: if provide will not load from data_path

        Returns: 
            None

        """
        # get data
        dataloader = self.datamodule.online_dataloader(data)
        result = []
        with torch.no_grad():
            for batch in dataloader:
                result.append(self.imodel.predict_step(batch))
        # start predict
        return self.imodel.postprocessor(stage='predict',
                                    list_batch_outputs=result,
                                    origin_data=data,
                                    rt_config={},
                                    save_condition=False)
