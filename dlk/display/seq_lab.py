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

from dlk.display import Display, DisplayConfigBase
from dlk.utils.display.ner import NerVisualizer
from dlk.utils.register import register

logger = logging.getLogger(__name__)


class SeqLabDisplayConfig(DisplayConfigBase):
    """the seq_lab display"""

    title = StrField("Sequence Labeling Task Demo", help="the title of the demo")
    help = StrField(
        "You can add the sentence to the `sentence input area` and click `submit` to get the entities info of this sentence",
        help="the help infomation of the demo",
    )

    render_height = IntField(value=500, help="the display height")
    render_width = IntField(value=900, help="the display width")
    render_type = StrField(value="html", help="the render type of the display")
    output_head = StrField(value="Sequence Label", help="the output head")

    class Input:
        sentence = StrField(
            "text",
            options=["text"],
            help="the input values name for seq lab, the value is the type",
        )

    input = NestField(value=Input, help="the input config")

    class Hold:
        entities_info = ListField([], help="the default value of entities_info")

    hold = NestField(value=Hold, help="the hold values")

    ignore_labels = ListField(
        value=["O", "X", "S", "E"],
        help="the ignore labels, if the entity label in this list, we will ignore this entity",
    )


cregister("display", "seq_lab")(SeqLabDisplayConfig)
cregister("display", "span_cls")(SeqLabDisplayConfig)


class SeqLabDisplay(Display):
    """sequence labeling display"""

    def __init__(self, config: SeqLabDisplayConfig):
        super(SeqLabDisplay, self).__init__(config)
        self.visual = NerVisualizer(config.ignore_labels)

    def display(self, result):
        return self.visual.display(result)


register("display", "seq_lab")(SeqLabDisplay)
register("display", "span_cls")(SeqLabDisplay)
