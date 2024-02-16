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

from dlk.display.summary import DisplayConfigBase, SummaryDisplay, SummaryDisplayConfig
from dlk.utils.register import register

logger = logging.getLogger(__name__)


@cregister("display", "img_caption")
class ImageCaptionDisplayConfig(DisplayConfigBase):
    """the image caption display"""

    title = StrField("Image Caption Task Demo", help="the title of the demo")
    help = StrField(
        "You can add the image to the `image area` and click `submit` to get the caption text of the image",
        help="the help infomation of the demo",
    )
    render_type = StrField(value="text", help="the output type of is text")
    output_head = StrField(value="Image Caption", help="the output head")

    class Input:
        image = StrField(
            value="image",
            help="the input name for img_caption, the value is the input type",
        )

    input = NestField(value=Input, help="the input config")

    class Hold:
        target = StrField("", options=[""], help="the default target of img_caption")

    hold = NestField(value=Hold, help="the hold values")


@register("display", "img_caption")
class ImageCaptionDisplay(SummaryDisplay):
    """text classification display"""

    def __init__(self, config: ImageCaptionDisplayConfig):
        super(ImageCaptionDisplay, self).__init__(config)
