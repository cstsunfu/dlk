# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Callable, Dict, List, Union

import pandas as pd

from dlk.online import OnlinePredict
from dlk.preprocess import PreProcessor

logger = logging.getLogger(__name__)


class Server(object):
    """Demo"""

    def __init__(
        self,
        process_config: Union[str, Dict] = "/path/to/config",
        fit_config: Union[str, Dict] = "/path/to/config",
        checkpoint: str = "/path/to/checkpoint",
    ):
        super(Server, self).__init__()
        self.processor = self.get_processor(process_config)
        self.predictor = self.get_predictor(fit_config, checkpoint)

    def fit(self, input_df: pd.DataFrame) -> Dict:
        """use the model to get the result of the data want to predict

        Args:
            input_df: the one input data, it could be only one row or multiple rows(batch)

        Returns:
            the prediction result

        """
        processed_data = self.processor.fit(input_df)
        result = self.predictor.predict.predict(processed_data)
        return result

    def get_processor(self, config: Union[str, Dict]) -> PreProcessor:
        """get the preprocessor instance

        Args:
            config: the path to config or the confic dict

        Returns:
            PreProcessor instance

        """
        return PreProcessor(config, stage="online")

    def get_predictor(self, config: Union[str, Dict], checkpoint) -> OnlinePredict:
        """get the preprocessor instance

        Args:
            config: the path to config or the confic dict

        Returns:
            PreProcessor instance

        """
        return OnlinePredict(config, checkpoint)
