# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os

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

from dlk.nn.base_module import BaseIdentityModule
from dlk.utils.import_module import import_module_dir
from dlk.utils.register import register


@cregister("encoder", "identity")
class IdentityModuleConfig:
    """identity encoder"""

    pass


@register("encoder", "identity")
class IdentityModule(BaseIdentityModule):
    pass


encoder_dir = os.path.dirname(__file__)
import_module_dir(encoder_dir, "dlk.nn.layer.encoder")
