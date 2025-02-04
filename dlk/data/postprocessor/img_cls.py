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
