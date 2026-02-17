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

    def __init__(self, config: Union[str, dict], checkpoint: str, update_config=None):
        super(OnlinePredict, self).__init__(config, checkpoint, update_config)
        datamodule, _ = self.get_datamodule(
            self.dlk_config, {}, world_size=self.trainer.world_size
        )
        self.online = True
        self.datamodule = datamodule

    def predict(self, data):
        dataloader = self.datamodule.online_dataloader(data)
        result = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                out = self.imodel.predict_step(batch, i)
                out = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in out.items()
                }
                result.append(out)

        # Delegate correctly to pipeline __call__ method
        return self.imodel.postprocessor(
            stage="online",
            list_batch_outputs=result,
            origin_data=data,
            rt_config={},
            save_condition=False,
        )
