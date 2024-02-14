# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Dict, Tuple

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


@cregister("token_gen_decoder", "identity")
class IdentityModuleConfig:
    """identity decoder"""

    pass


@register("token_gen_decoder", "identity")
class IdentityModule(BaseIdentityModule):
    pass


decoder_dir = os.path.dirname(__file__)
import_module_dir(decoder_dir, "dlk.nn.layer.token_gen_decoder")
