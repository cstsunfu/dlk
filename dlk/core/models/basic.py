from . import model_register, model_config_register
from typing import Dict, List
from dlk.utils.config import BaseConfig, ConfigTool
from dlk.core.base_module import BaseModel
import torch
from dlk.core.layers.embeddings import embedding_config_register, embedding_register
from dlk.core.initmethods import initmethod_config_register, initmethod_register
from dlk.core.layers.encoders import encoder_config_register, encoder_register
from dlk.core.layers.decoders import decoder_config_register, decoder_register


@model_config_register('basic')
class BasicModelConfig(BaseConfig):
    """docstring for BasicIModelConfig
    {
        embedding: {
            _base: "static"
            config: {
                embedding_file: "*@*", //the embedding file, must be saved as numpy array by pickle
                embedding_dim: "*@*",
                //if the embedding_file is a dict, you should provide the dict trace to embedding
                embedding_trace: ".", //default the file itself is the embedding
                /*embedding_trace: "embedding", //this means the <embedding = pickle.load(embedding_file)["embedding"]>*/
                /*embedding_trace: "meta.embedding", //this means the <embedding = pickle.load(embedding_file)['meta']["embedding"]>*/
                freeze: false, // is freeze
                dropout: 0, //dropout rate
                output_map: {},
            },
        },
        decoder: {
            _base: "linear",
            config: {
                input_size: "*@*",
                output_size: "*@*",
                pool: null,
                dropout: "*@*", //the decoder output no need dropout
                output_map: {}
            },
        },
        encoder: {
            _base: "lstm",
            config: {
                output_map: {},
                hidden_size: "*@*",
                input_size: *@*,
                output_size: "*@*",
                num_layers: 1,
                dropout: "*@*", // dropout between layers
            },
        },
        "initmethod": {
            "_base": "range_norm"
        },
        "config": {
            "embedding_dim": "*@*",
            "dropout": "*@*",
            "embedding_file": "*@*",
            "embedding_trace": "token_embedding",
        },
        _link: {
            "config.embedding_dim": ["embedding.config.embedding_dim",
                                     "encoder.config.input_size",
                                     "encoder.config.output_size",
                                     "encoder.config.hidden_size",
                                     "decoder.config.output_size",
                                     "decoder.config.input_size"
                                    ],
            "config.dropout": ["encoder.config.dropout", "decoder.config.dropout", "embedding.config.dropout"],
            "config.embedding_file": ['embedding.config.embedding_file'],
            "config.embedding_trace": ['embedding.config.embedding_trace']
        }
        _name: "basic"
    }
    """
    def __init__(self, config):
        super(BasicModelConfig, self).__init__(config)

        self.embedding, self.embedding_config = self.get_embedding(config["embedding"])
        self.encoder, self.encoder_config = self.get_encoder(config["encoder"])
        self.decoder, self.decoder_config = self.get_decoder(config["decoder"])
        self.init_method, self.init_method_config = self.get_init_method(config["initmethod"])
        self.config = config['config']

    def get_embedding(self, config):
        """get embedding config and embedding module

        :config: TODO
        :returns: TODO

        """
        return ConfigTool.get_leaf_module(embedding_register, embedding_config_register, "embedding", config)

    def get_init_method(self, config):
        """get embedding config and embedding module

        :config: TODO
        :returns: TODO

        """
        return ConfigTool.get_leaf_module(initmethod_register, initmethod_config_register, "init method", config)

    def get_encoder(self, config):
        """get encoder config and encoder module

        :config: TODO
        :returns: TODO

        """
        return ConfigTool.get_leaf_module(encoder_register, encoder_config_register, "encoder", config)

    def get_decoder(self, config):
        """get decoder config and decoder module

        :config: TODO
        :returns: TODO
        """
        return ConfigTool.get_leaf_module(decoder_register, decoder_config_register, "decoder", config)


@model_register('basic')
class BasicModel(BaseModel):
    """
    Sequence labeling model
    """

    def __init__(self, config: BasicModelConfig, checkpoint):
        super().__init__()

        self.embedding = config.embedding(config.embedding_config)
        self.encoder = config.encoder(config.encoder_config)
        self.decoder = config.decoder(config.decoder_config)

        if not checkpoint:
            init_method = config.init_method(config.init_method_config)
            self.embedding.init_weight(init_method)
            self.encoder.init_weight(init_method)
            self.decoder.init_weight(init_method)

        self.config = config.config
        self._provided_keys = self.config.get("provided_keys", [])

    def provide_keys(self)->List[str]:
        """TODO: should provide_keys in model?
        """
        return self.decoder.provided_keys()

    def check_keys_are_provided(self, provide: List[str]=[])->None:
        """TODO: should check keys in model?
        """
        self._provided_keys = self._provided_keys + provide
        self.embedding.check_keys_are_provided(self._provided_keys)
        self.encoder.check_keys_are_provided(self.embedding.provide_keys())
        self.decoder.check_keys_are_provided(self.encoder.provide_keys())

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """forward
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        embedding_outputs = self.embedding(inputs)
        encode_outputs = self.encoder(embedding_outputs)
        decode_outputs = self.decoder(encode_outputs)
        return decode_outputs

    def predict_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """predict
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        embedding_outputs = self.embedding.predict_step(inputs)
        encode_outputs = self.encoder.predict_step(embedding_outputs)
        decode_outputs = self.decoder.predict_step(encode_outputs)
        return decode_outputs

    def training_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """training
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        embedding_outputs = self.embedding.training_step(inputs)
        encode_outputs = self.encoder.training_step(embedding_outputs)
        decode_outputs = self.decoder.training_step(encode_outputs)
        return decode_outputs

    def validation_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """valid
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        embedding_outputs = self.embedding.validation_step(inputs)
        encode_outputs = self.encoder.validation_step(embedding_outputs)
        decode_outputs = self.decoder.validation_step(encode_outputs)
        return decode_outputs

    def test_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """valid
        :inputs: Dict[str: torch.Tensor], one mini-batch inputs
        :returns: Dict[str: torch.Tensor], one mini-batch outputs
        """
        embedding_outputs = self.embedding.test_step(inputs)
        encode_outputs = self.encoder.test_step(embedding_outputs)
        decode_outputs = self.decoder.test_step(encode_outputs)
        return decode_outputs
