# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Callable, Dict, List, Union

import torch

from dlk.predict import Predict

logger = logging.getLogger(__name__)


class OnlinePredict(Predict):
    """OnlinePredict"""

    def __init__(self, config: Union[str, dict], checkpoint: str):
        super(OnlinePredict, self).__init__(config, checkpoint)
        datamodule, _ = self.get_datamodule(
            self.dlk_config, {}, world_size=self.trainer.world_size
        )
        self.online = True
        self.datamodule = datamodule

    def predict(self, data):
        """init the model, datamodule, manager then predict the predict_dataloader

        Args:
            data: the preprocessed data

        Returns:
            None

        """
        # get data
        dataloader = self.datamodule.online_dataloader(data)
        result = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                result.append(self.imodel.predict_step(batch, i))
        # start predict
        return self.imodel.postprocessor(
            stage="online",
            list_batch_outputs=result,
            origin_data=data,
            rt_config={},
            save_condition=False,
        )
