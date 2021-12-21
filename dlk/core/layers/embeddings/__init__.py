# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""embeddings"""
import importlib
import os
from typing import Callable, Dict, Tuple, Any
from dlk.utils.register import Register

embedding_config_register = Register("Embedding config register.")
embedding_register = Register("Embedding register.")


def import_embeddings(embeddings_dir, namespace):
    for file in os.listdir(embeddings_dir):
        path = os.path.join(embeddings_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            embedding_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + embedding_name)


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


# automatically import any Python files in the embeddings directory
embeddings_dir = os.path.dirname(__file__)
import_embeddings(embeddings_dir, "dlk.core.layers.embeddings")
