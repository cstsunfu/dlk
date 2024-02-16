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
from pyecharts.charts import Gauge
from pyecharts.options import InitOpts

from dlk.display import Display, DisplayConfigBase
from dlk.utils.register import register

logger = logging.getLogger(__name__)


@cregister("display", "txt_reg")
class TextRegDisplayConfig(DisplayConfigBase):
    """the text classification display"""

    title = StrField("Text Regression Task Demo", help="the title of the demo")
    help = StrField(
        "You can add the sentence to the `sentence input area` and click `submit` to get the regression info of this sentence",
        help="the help infomation of the demo",
    )
    render_height = IntField(value=300, help="the display height")
    render_width = IntField(value=500, help="the display width")
    render_type = StrField(value="html", help="the render type of the display")
    output_head = StrField(value="Logic Regression Score", help="the output head")

    class Input:
        sentence = StrField(
            "text",
            options=["text"],
            help="the input values name for text regression, the value is the type",
        )

    input = NestField(value=Input, help="the input config")


@cregister("display", "txt_match_reg")
class TextMatchRegDisplayConfig(DisplayConfigBase):
    """the text match display"""

    title = StrField("Text Match Regression Task Demo", help="the title of the demo")
    help = StrField(
        "You can add the sentence(a and b) to the `sentence input area(a and b)` and click `submit` to get the match regression info of the pair sentences",
        help="the help infomation of the demo",
    )

    class Input:
        sentence_a = StrField("text", options=["text"], help="the text a for match")
        sentence_b = StrField("text", options=["text"], help="the text b for match")

    render_height = IntField(value=300, help="the display height")
    render_width = IntField(value=500, help="the display width")

    input = NestField(value=Input, help="the input config")


class TextRegDisplay(Display):
    """text classification display"""

    def __init__(self, config: TextRegDisplayConfig):
        super(TextRegDisplay, self).__init__(config)
        self.config = config
        self.init_options = InitOpts(
            width=f"{config.render_width}px", height=f"{config.render_height}px"
        )

    def display(self, result):
        predict = result["predict_values"][0] * 100
        return (
            Gauge(init_opts=self.init_options)
            .add(series_name="Regression", data_pair=[["", 55.5]])
            .render_embed()
        )


register("display", "txt_reg")(TextRegDisplay)
register("display", "txt_match_reg")(TextRegDisplay)
