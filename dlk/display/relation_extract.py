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
from dlk.utils.display.relation_extraction import RelationExtractionVisualizer
from dlk.utils.register import register

logger = logging.getLogger(__name__)


@cregister("display", "span_relation")
class RelationDisplayConfig(DisplayConfigBase):
    """the relation extraction display"""

    title = StrField("Relation Extraction Task Demo", help="the title of the demo")
    help = StrField(
        "You can add the sentence to the `sentence input area` and click `submit` to get the relations info of this sentence",
        help="the help infomation of the demo",
    )
    render_height = IntField(value=500, help="the display height")
    render_width = IntField(value=900, help="the display width")
    render_type = StrField(value="html", help="the render type of the display")

    output_head = StrField(value="Relations", help="the output head")

    class Input:
        sentence = StrField(
            "text",
            options=["text"],
            help="the input values name of relations extraction, the value is the type",
        )

    input = NestField(value=Input, help="the input config")

    class Hold:
        entities_info = ListField([], help="the value of entities_info")
        relations_info = ListField([], help="the value of relations_info")

    hold = NestField(value=Hold, help="the hold values")

    ignore_relations = ListField(
        value=["O"],
        help="the relation types which is ignored for display",
    )


@register("display", "span_relation")
class RelationDisplay(Display):
    """relation extraction display"""

    def __init__(self, config: RelationDisplayConfig):
        super(RelationDisplay, self).__init__(config)
        self.visual = RelationExtractionVisualizer(
            config.ignore_relations, width=config.render_width
        )

    def display(self, result):
        return self.visual.display(result)
