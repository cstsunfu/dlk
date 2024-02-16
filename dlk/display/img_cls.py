# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import pickle as pkl
from typing import Dict, List, Optional, Tuple, Union

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
from pyecharts.charts import Bar
from pyecharts.options import InitOpts

from dlk.display.txt_cls import (
    TextClassificationDisplay,
    TextClassificationDisplayConfig,
)
from dlk.utils.register import register

logger = logging.getLogger(__name__)


@cregister("display", "img_cls")
class ImageClassificationDisplayConfig(TextClassificationDisplayConfig):
    """the image classification display"""

    title = StrField("Image Classification Task Demo", help="the title of the demo")
    help = StrField(
        "You can upload the image to the `image upload` and click `submit` to get the class info of this sentence",
        help="the help infomation of the demo",
    )
    output_head = StrField(value="Image Class", help="the output head")

    class Input:
        image = StrField(
            "image",
            options=["image"],
            help="the input values name for classification, the value is the type",
        )

    input = NestField(value=Input, help="the input config")


@register("display", "img_cls")
class ImageClassificationDisplay(TextClassificationDisplay):
    """image classification display"""

    def __init__(self, config: ImageClassificationDisplayConfig):
        super(ImageClassificationDisplay, self).__init__(config)
        self.config = config
