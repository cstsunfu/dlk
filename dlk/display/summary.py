# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging

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

from dlk.display import Display, DisplayConfigBase
from dlk.utils.register import register

logger = logging.getLogger(__name__)


@cregister("display", "summary")
class SummaryDisplayConfig(DisplayConfigBase):
    """the summary display"""

    title = StrField("Text Summary Task Demo", help="the title of the demo")
    help = StrField(
        "You can add the origin input to the `input area` and click `submit` to get the summary text of the input",
        help="the help infomation of the demo",
    )
    render_height = IntField(value=200, help="the display height")
    render_width = IntField(value=600, help="the display width")
    render_type = StrField(value="text", help="the render type of the display")
    output_head = StrField(value="Summary", help="the output head")

    class Input:
        input = StrField(
            value="text",
            options=["text"],
            help="the input values name for summary, the value is the input type",
        )

    input = NestField(value=Input, help="the input config")

    class Hold:
        target = StrField("", options=[""], help="the default target of summary")

    hold = NestField(value=Hold, help="the hold values")


@register("display", "summary")
class SummaryDisplay(Display):
    """text classification display"""

    def __init__(self, config: SummaryDisplayConfig):
        super(SummaryDisplay, self).__init__(config)

    def display(self, result):
        generate_summary = result["generated"][0]["generate"]
        return generate_summary
