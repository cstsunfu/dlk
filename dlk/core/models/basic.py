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

from typing import Dict, List
from dlk.core.base_module import BaseModel
import torch
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules, ConfigTool


@config_register("model", "basic")
@define
class BasicModelConfig(BaseConfig):
    name = NameField(value="basic", file=__file__, help="the basic model")
    config = DictField(value={}, help="default null config for basic model")
    submods = SubModules(value={
        "embedding": "identity",
        "encoder": "identity",
        "decoder": "identity",
        "initmethod": "default",
        }, help="submodules for basic model")

@register("model", 'basic')
class BasicModeConfig(BaseConfig):
    """Config for BasicModel

    Config Example:
        >>> {
        >>>     embedding: {
        >>>         _base: "static"
        >>>         config: {
        >>>             embedding_file: "*@*", //the embedding file, must be saved as numpy array by pickle
        >>>             embedding_dim: "*@*",
        >>>             //if the embedding_file is a dict, you should provide the dict trace to embedding
        >>>             embedding_trace: ".", //default the file itself is the embedding
        >>>             /*embedding_trace: "embedding", //this means the <embedding = pickle.load(embedding_file)["embedding"]>*/
        >>>             /*embedding_trace: "meta.embedding", //this means the <embedding = pickle.load(embedding_file)['meta']["embedding"]>*/
        >>>             freeze: false, // is freeze
        >>>             dropout: 0, //dropout rate
        >>>             output_map: {},
        >>>         },
        >>>     },
        >>>     decoder: {
        >>>         _base: "linear",
        >>>         config: {
        >>>             input_size: "*@*",
        >>>             output_size: "*@*",
        >>>             pool: null,
        >>>             dropout: "*@*", //the decoder output no need dropout
        >>>             output_map: {}
        >>>         },
        >>>     },
        >>>     encoder: {
        >>>         _base: "lstm",
        >>>         config: {
        >>>             output_map: {},
        >>>             hidden_size: "*@*",
        >>>             input_size: *@*,
        >>>             output_size: "*@*",
        >>>             num_layers: 1,
        >>>             dropout: "*@*", // dropout between layers
        >>>         },
        >>>     },
        >>>     "initmethod": {
        >>>         "_base": "range_norm"
        >>>     },
        >>>     "config": {
        >>>         "embedding_dim": "*@*",
        >>>         "dropout": "*@*",
        >>>         "embedding_file": "*@*",
        >>>         "embedding_trace": "token_embedding",
        >>>     },
        >>>     _link: {
        >>>         "config.embedding_dim": ["embedding.config.embedding_dim",
        >>>                                  "encoder.config.input_size",
        >>>                                  "encoder.config.output_size",
        >>>                                  "encoder.config.hidden_size",
        >>>                                  "decoder.config.output_size",
        >>>                                  "decoder.config.input_size"
        >>>                                 ],
        >>>         "config.dropout": ["encoder.config.dropout", "decoder.config.dropout", "embedding.config.dropout"],
        >>>         "config.embedding_file": ['embedding.config.embedding_file'],
        >>>         "config.embedding_trace": ['embedding.config.embedding_trace']
        >>>     }
        >>>     _name: "basic"
        >>> }
    """
    def __init__(self, config):
        super(BasicModelConfig, self).__init__(config)

        self.embedding, self.embedding_config = self.get_embedding(config["embedding"])
        self.encoder, self.encoder_config = self.get_encoder(config["encoder"])
        self.decoder, self.decoder_config = self.get_decoder(config["decoder"])
        self.init_method, self.init_method_config = self.get_init_method(config["initmethod"])
        self.config = config['config']

    def get_embedding(self, config: Dict):
        """return the Embedding and EmbeddingConfig

        Args:
            config: the embedding config

        Returns: 
            Embedding, EmbeddingConfig

        """
        return ConfigTool.get_leaf_module(embedding_register, embedding_config_register, "embedding", config)

    def get_init_method(self, config: Dict):
        """return the InitMethod and InitMethodConfig

        Args:
            config: the init method config

        Returns: 
            InitMethod, InitMethodConfig

        """
        return ConfigTool.get_leaf_module(initmethod_register, initmethod_config_register, "init method", config)

    def get_encoder(self, config: Dict):
        """return the Encoder and EncoderConfig

        Args:
            config: the encoder config

        Returns: 
            Encoder, EncoderConfig

        """
        return ConfigTool.get_leaf_module(encoder_register, encoder_config_register, "encoder", config)

    def get_decoder(self, config):
        """return the Decoder and DecoderConfig

        Args:
            config: the decoder config

        Returns: 
            Decoder, DecoderConfig

        """
        return ConfigTool.get_leaf_module(decoder_register, decoder_config_register, "decoder", config)


@register("model", 'basic')
class BasicModel(BaseModel):
    """Basic & General Model
    """
    def __init__(self, config: BasicModelConfig, checkpoint):
        super().__init__()
        dict_config = config.to_dict()
        self.embedding = ConfigTool.get_leaf_module(register, config_register, "embedding", dict_config['embedding'], init=True)
        self.encoder = ConfigTool.get_leaf_module(register, config_register, "encoder", dict_config['encoder'], init=True)
        self.decoder = ConfigTool.get_leaf_module(register, config_register, "decoder", dict_config['decoder'], init=True)

        if not checkpoint:
            init_method = ConfigTool.get_leaf_module(register, config_register, "initmethod", dict_config['initmethod'], init=True)
            self.embedding.init_weight(init_method)
            self.encoder.init_weight(init_method)
            self.decoder.init_weight(init_method)

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns: the outputs

        """
        embedding_outputs = self.embedding(inputs)
        encode_outputs = self.encoder(embedding_outputs)
        decode_outputs = self.decoder(encode_outputs)
        return decode_outputs

    def predict_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do predict for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: the predicts outputs

        """
        embedding_outputs = self.embedding.predict_step(inputs)
        encode_outputs = self.encoder.predict_step(embedding_outputs)
        decode_outputs = self.decoder.predict_step(encode_outputs)
        return decode_outputs

    def training_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do training for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: the training outputs

        """
        embedding_outputs = self.embedding.training_step(inputs)
        encode_outputs = self.encoder.training_step(embedding_outputs)
        decode_outputs = self.decoder.training_step(encode_outputs)
        return decode_outputs

    def validation_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do validation for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: the validation outputs

        """
        embedding_outputs = self.embedding.validation_step(inputs)
        encode_outputs = self.encoder.validation_step(embedding_outputs)
        decode_outputs = self.decoder.validation_step(encode_outputs)
        return decode_outputs

    def test_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do test for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: the test outputs

        """
        embedding_outputs = self.embedding.test_step(inputs)
        encode_outputs = self.encoder.test_step(embedding_outputs)
        decode_outputs = self.decoder.test_step(encode_outputs)
        return decode_outputs
