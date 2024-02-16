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
from pyecharts.options import (
    AxisOpts,
    InitOpts,
    LabelOpts,
    MarkPointItem,
    MarkPointOpts,
)

from dlk.display import Display, DisplayConfigBase
from dlk.utils.register import register

logger = logging.getLogger(__name__)


@cregister("display", "txt_cls")
class TextClassificationDisplayConfig(DisplayConfigBase):
    """the text classification display"""

    title = StrField("Text Classification Task Demo", help="the title of the demo")
    help = StrField(
        "You can add the sentence to the `sentence input area` and click `submit` to get the class info of this sentence",
        help="the help infomation of the demo",
    )
    render_height = IntField(value=500, help="the display height")
    render_width = IntField(value=900, help="the display width")
    render_type = StrField(value="html", help="the render type of the display")
    output_head = StrField(value="Text Class", help="the output head")

    class Input:
        sentence = StrField(
            "text",
            options=["text"],
            help="the input values name for classification, the value is the type",
        )

    labels = ListField(
        [], help=f"the display labels, if not set, will display all the labels"
    )
    top_k = IntField(
        10, minimum=0, help="the top k class to display, if 0, will display all"
    )

    input = NestField(value=Input, help="the input config")


@cregister("display", "txt_match")
class TextMatchDisplayConfig(TextClassificationDisplayConfig):
    """the text match display"""

    title = StrField("Text Match Task Demo", help="the title of the demo")
    help = StrField(
        "You can add the sentence(a and b) to the `sentence input area(a and b)` and click `submit` to get the match class info of the pair sentences",
        help="the help infomation of the demo",
    )

    class Input:
        sentence_a = StrField("text", options=["text"], help="the text a for match")
        sentence_b = StrField("text", options=["text"], help="the text b for match")

    input = NestField(value=Input, help="the input config")


class TextClassificationDisplay(Display):
    """text classification display"""

    def __init__(self, config: TextClassificationDisplayConfig):
        super(TextClassificationDisplay, self).__init__(config)
        self.config = config
        self.init_options = InitOpts(
            width=f"{config.render_width}px",
            height=f"{config.render_height}px",
        )

    def display(self, result):
        predicts = result["predicts"]
        label_values = {}
        for _, pair in predicts.items():
            label_values[pair[0]] = float(f"{pair[1]*100:.2f}")
        if self.config.top_k > 0:
            label_values = dict(
                sorted(label_values.items(), key=lambda x: x[1], reverse=True)[
                    : self.config.top_k
                ]
            )
        labels = []
        if self.config.labels:
            labels = self.config.labels
        else:
            labels = list(label_values.keys())
            labels.sort()
        return (
            Bar(init_opts=self.init_options)
            .set_global_opts(
                yaxis_opts=AxisOpts(
                    axislabel_opts=LabelOpts(formatter="{value} %"),
                    name="Probability (%)",
                ),
                xaxis_opts=AxisOpts(name="Class", axislabel_opts=LabelOpts(rotate=45)),
            )
            .add_xaxis(labels)
            .add_yaxis(
                "Probability",
                [label_values[label] for label in labels],
                category_gap="50%",
            )
            .set_series_opts(
                label_opts=LabelOpts(is_show=False),
                markpoint_opts=MarkPointOpts(
                    data=[
                        MarkPointItem(type_="max", name="Top 1"),
                    ]
                ),
            )
            .render_embed()
        )


register("display", "txt_cls")(TextClassificationDisplay)
register("display", "txt_match")(TextClassificationDisplay)
