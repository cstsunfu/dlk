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

import torch.nn as nn
from . import embedding_register, embedding_config_register
from typing import Dict, List, Set
from dlk.core.base_module import SimpleModule, BaseModuleConfig
import pickle as pkl
from dlk.utils.logger import Logger
import torch

logger = Logger.get_logger()

@embedding_config_register('shared_static')
class SharedStaticEmbeddingConfig(BaseModuleConfig):
    """Config for SharedStaticEmbedding

    Config Example:
        >>> {
        >>>     "config": {
        >>>         "embedding_file": "*@*", //the embedding file, must be saved as numpy array by pickle
        >>>         "embedding_dim": "*@*",
        >>>         //if the embedding_file is a dict, you should provide the dict trace to embedding
        >>>         "embedding_trace": ".", //default the file itself is the embedding
        >>>         /*embedding_trace: "embedding", //this means the <embedding = pickle.load(embedding_file)["embedding"]>*/
        >>>         /*embedding_trace: "meta.embedding", //this means the <embedding = pickle.load(embedding_file)['meta']["embedding"]>*/
        >>>         "freeze": false, // is freeze
        >>>         "padding_idx": 0, //dropout rate
        >>>         "dropout": 0, //dropout rate
        >>>         "output_map": {},
        >>>         "input_map": {}, // required_key: provide_key
        >>>     },
        >>>     "_name": "shared_static",
        >>> }
    """
    def __init__(self, config: Dict):
        super(SharedStaticEmbeddingConfig, self).__init__(config)
        config = config['config']

        embedding_file = config['embedding_file']
        embedding_file = pkl.load(open(embedding_file, 'rb'))
        embedding_trace = config["embedding_trace"]
        if embedding_trace != '.':
            traces = embedding_trace.split('.')
            for trace in traces:
                embedding_file = embedding_file[trace]
        self.embedding = embedding_file
        self.freeze = config['freeze']
        self.dropout = config['dropout']
        self.padding_idx = config['padding_idx']
        self.embedding_dim = config['embedding_dim']
        self.post_check(config, used=[
            "embedding_file",
            "embedding_dim",
            "embedding_trace",
            "freeze",
            "padding_idx",
            "dropout"
        ])


@embedding_register('shared_static')
class SharedStaticEmbedding(SimpleModule):
    """ from 'input_ids' generate shared_static 'embedding' like glove, word2vec
    """

    def __init__(self, config: SharedStaticEmbeddingConfig):
        super().__init__(config)
        self._provided_keys = set() # provided by privous module, will update by the check_keys_are_provided
        self._provide_keys = {'embedding'} # provide by this module
        self._required_keys = {'input_ids', 'output_ids'} # required by this module
        self.config = config
        self.dropout = nn.Dropout(float(self.config.dropout))
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(self.config.embedding, dtype=torch.float), freeze=self.config.freeze, padding_idx=self.config.padding_idx)
        assert self.embedding.weight.shape[-1] == self.config.embedding_dim

    def init_weight(self, method):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns: 
            None

        """
        logger.info(f'The shared_static embedding is loaded the pretrained.')

    def forward(self, inputs: Dict[str, torch.Tensor], decode=False)->Dict[str, torch.Tensor]:
        """get the pretrained shared_static embedding like glove word2vec

        Args:
            inputs: one mini-batch inputs
            decode: the decode embedding or the encode embedding

        Returns: 
            one mini-batch outputs

        """
        if decode:
            inputs[self.get_output_name('decode_embedding')] = self.dropout(self.embedding(inputs[self.get_input_name('output_ids')]))
            # WARNING never gather the decode embedding
            # if self._logits_gather.layer_map:
                # inputs.update(self._logits_gather([inputs[self.get_output_name('decode_embedding')]]))
        else:
            inputs[self.get_output_name('embedding')] = self.dropout(self.embedding(inputs[self.get_input_name('input_ids')]))
            if self._logits_gather.layer_map:
                inputs.update(self._logits_gather([inputs[self.get_output_name('embedding')]]))

        return inputs

    def predict_step(self, inputs: Dict[str, torch.Tensor], decode: bool=False)->Dict[str, torch.Tensor]:
        """do predict for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        return self(inputs, decode)

    def training_step(self, inputs: Dict[str, torch.Tensor], decode: bool=False)->Dict[str, torch.Tensor]:
        """do train for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        return self(inputs, decode)

    def validation_step(self, inputs: Dict[str, torch.Tensor], decode: bool=False)->Dict[str, torch.Tensor]:
        """do validation for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        return self(inputs, decode)

    def test_step(self, inputs: Dict[str, torch.Tensor], decode: bool=False)->Dict[str, torch.Tensor]:
        """do test for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        return self(inputs, decode)
