# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from functools import partial
from typing import Callable, Dict, Iterable, List, Set, Union

import numpy as np
import pandas as pd
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
from tokenizers import Tokenizer
from transformers import ViTImageProcessor

from dlk.data.subprocessor import BaseSubProcessor, BaseSubProcessorConfig
from dlk.utils.io import open
from dlk.utils.register import register

logger = logging.getLogger(__name__)

hf_image_processor = {"vit": ViTImageProcessor}


@cregister("subprocessor", "image_process")
class ImageProcessConfig(BaseSubProcessorConfig):
    """the token norm subprocessor"""

    train_data_set = ListField(
        value=["train", "valid", "test"],
        suggestions=[["train", "valid", "test"]],
        help="the data set should be processed for train stage",
    )
    predict_data_set = ListField(
        value=["predict"],
        suggestions=[["predict"]],
        help="the data set should be processed for predict stage",
    )
    online_data_set = ListField(
        value=["online"],
        suggestions=[["online"]],
        help="the data set should be processed for online stage",
    )

    preprocess_config = StrField(
        value=MISSING,
        suggestions=["preprocess_config.json"],
        help="the hf image preprocess config path",
    )
    preprocess_method = StrField(
        value="vit",
        options=list(hf_image_processor.keys()),
        help="the hf image preprocess method",
    )

    class InputMap:
        image = StrField(
            value="image",
            suggestions=["image"],
            help="the input image",
        )

    input_map = NestField(
        value=InputMap,
        help="the input map of the processor, the key is the name of the processor needed key, the value is the provided data provided key",
    )

    class OutputMap:
        pixel_values = StrField(
            value="pixel_values",
            suggestions=["pixel_values"],
            help="the processed image values",
        )

    output_map = NestField(
        value=OutputMap,
        help="the output map of the processor, the key is the name of the processor provided key, the value is the nexted processor needed key",
    )


@register("subprocessor", "image_process")
class ImageProcess(BaseSubProcessor):
    """preprocess image"""

    def __init__(self, stage: str, config: ImageProcessConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.stage = stage
        self.config = config

        self.image_processor = hf_image_processor[
            self.config.preprocess_method
        ].from_json_file(self.config.preprocess_config)

    def image_process(self, input: pd.Series) -> np.ndarray:
        """norm token, the result len(result) == len(token), exp.  12348->00000

        Args:
            input:  include image

        Returns:
            normed_token

        """
        return self.image_processor(
            input[self.config.input_map.image], return_tensors="np"
        ).pixel_values

    def process(self, data: pd.DataFrame, deliver_meta: bool) -> pd.DataFrame:
        """image process entry

        Args:
            data:
            >>> |image                                           |
            >>> |------------------------------------------------|
            >>> |PIL.JpegImagePlugin.JpegImageFile image mode=...|
            >>> |PIL.JpegImagePlugin.JpegImageFile image mode=...|

            deliver_meta:
                False
        Returns:
            processed data

        """
        data[self.config.output_map.pixel_values] = data.apply(
            self.image_process, axis=1
        )

        return data
