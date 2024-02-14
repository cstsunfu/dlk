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


@cregister("embedding", "identity")
class IdentityModuleConfig:
    """identity embedding"""

    pass


@register("embedding", "identity")
class IdentityModule(BaseIdentityModule):
    pass


class EmbeddingInput(object):
    """docstring for EmbeddingInput"""

    def __init__(self, **args):
        super(EmbeddingInput, self).__init__()
        self.input_ids = args.get("input_ids", None)


class EmbeddingOutput(object):
    """docstring for EmbeddingOutput"""

    def __init__(self, **args):
        super(EmbeddingOutput, self).__init__()
        self.represent = args.get("represent", None)


embedding_dir = os.path.dirname(__file__)
import_module_dir(embedding_dir, "dlk.nn.layer.embedding")
