# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict

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
    dataclass,
)

from dlk.utils.import_module import import_module_dir


@dataclass
class DisplayConfigBase(Base):
    """the display config"""

    class Input:
        pass

    class Hold:
        pass

    title = StrField("", help="the title of the demo")
    help = StrField("", help="the help infomation of the demo")
    input = NestField(value=Input, help="the input config")
    hold = NestField(value=Hold, help="the hold value")
    output_head = StrField(value="Output", help="the output head")
    render_height = IntField(value=500, help="the display height")
    render_width = IntField(value=900, help="the display width")
    render_type = StrField(
        value="html",
        options=["html", "image", "text"],
        help="the render type of the display",
    )


class Display(object):
    """Display Base"""

    def __init__(self, config: DisplayConfigBase):
        super(Display, self).__init__()
        self.config = config

    def display(self, result: Dict) -> str:
        """

        Args:
            result (Dict): TODO

        Returns: TODO

        """
        raise NotImplementedError

    def __call__(self, result: Dict) -> str:
        """display the result

        Args:
            result:

        Returns:
            html content

        """
        assert isinstance(result, dict), "result should be a dict"

        return self.display(result)


imodel_dir = os.path.dirname(__file__)
import_module_dir(imodel_dir, "dlk.display")
