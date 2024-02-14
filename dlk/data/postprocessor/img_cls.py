# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import pickle as pkl
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from intc import (
    MISSING,
    AnyField,
    Base,
    BoolField,
    DictField,
    FloatField,
    IntField,
    ListField,
    NestField,
    StrField,
    SubModule,
    cregister,
)

from dlk.data.postprocessor import BasePostProcessor, BasePostProcessorConfig
from dlk.data.postprocessor.txt_cls import (
    TxtClsPostProcessor,
    TxtClsPostProcessorConfig,
)
from dlk.utils.io import open
from dlk.utils.register import register
from dlk.utils.vocab import Vocabulary

logger = logging.getLogger(__name__)


@cregister("postprocessor", "img_cls")
class ImgClsPostProcessorConfig(TxtClsPostProcessorConfig):
    """image classfication postprocessor"""

    data_type = StrField(
        value="single", options=["single"], help="the data type, only single support"
    )

    class OriginInputMap:
        uuid = StrField(value="uuid", help="the uuid or the id of the sample")
        image = StrField(value="image", help="the image of the sample")
        image_url = StrField(value="", help="the url of the image, default is empty")

    origin_input_map = NestField(
        value=OriginInputMap,
        help="the origin input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )


@register("postprocessor", "img_cls")
class ImgClsPostProcessor(TxtClsPostProcessor):
    """postprocess for text classfication"""

    def __init__(self, config: ImgClsPostProcessorConfig):
        super(ImgClsPostProcessor, self).__init__(config)
        self.config = config

    def _get_origin_data(self, one_origin: pd.Series) -> Dict:
        """

        Args:
            one_origin: the original data

        Returns:
            the gather origin data

        """
        origin = {}
        if self.config.origin_input_map.image_url:
            origin["image_url"] = one_origin[self.config.origin_input_map.image_url]
        origin["uuid"] = one_origin[self.config.origin_input_map.uuid]
        return origin

    def do_save(
        self,
        predicts: List,
        stage: str,
        list_batch_outputs: List[Dict],
        origin_data: pd.DataFrame,
        rt_config: Dict,
        save_condition: bool = False,
    ):
        """save the predict when save_condition==True

        Args:
            predicts: list of predicts
            stage: train/test/etc.
            list_batch_outputs: a list of outputs
            origin_data: the origin pd.DataFrame data, there are some data not be able to convert to tensor
            rt_config:
                >>> current status
                >>> {
                >>>     "current_step": self.global_step,
                >>>     "current_epoch": self.current_epoch,
                >>>     "total_steps": self.num_training_steps,
                >>>     "total_epochs": self.num_training_epochs
                >>> }
            save_condition: True for save, False for depend on rt_config

        Returns:
            None

        """
        if self.config.start_save_epoch == -1 or self.config.start_save_step == -1:
            self.config.start_save_step = rt_config.get("total_steps", 0) - 1
            self.config.start_save_epoch = rt_config.get("total_epochs", 0) - 1
        if not save_condition and (
            rt_config["current_step"] >= self.config.start_save_step
            or rt_config["current_epoch"] >= self.config.start_save_epoch
        ):
            save_condition = True
        if save_condition:
            save_path = os.path.join(
                self.config.save_root_path, self.config.save_dir.get(stage, "")
            )
            if "current_step" in rt_config:
                save_file = os.path.join(
                    save_path, f"step_{str(rt_config['current_step'])}_predict.json"
                )
            else:
                save_file = os.path.join(save_path, "predict.json")
            logger.info(f"Save the {stage} predict data at {save_file}")
            with open(save_file, "w") as f:
                json.dump(predicts, f, indent=4, ensure_ascii=False)
