import torch.nn as nn
from . import model_register, model_config_register
from typing import Dict
from dlkit.utils.config import Config
# from dlkit.modules.embeddings import EMBEDDING_REGISTRY, EMBEDDING_CONFIG_REGISTRY
# from dlkit.modules.encoders import ENCODER_REGISTRY, ENCODER_CONFIG_REGISTRY
# from dlkit.modules.decoders import DECODER_REGISTRY, DECODER_CONFIG_REGISTRY
EMBEDDING_REGISTRY, EMBEDDING_CONFIG_REGISTRY, ENCODER_REGISTRY, ENCODER_CONFIG_REGISTRY, DECODER_REGISTRY, DECODER_CONFIG_REGISTRY = 1, 2, 3, 4, 5, 6

        
@model_config_register('basic')
class BasicModelConfig(Config):
    """docstring for BasicModelConfig"""
    def __init__(self, **kwargs):
        super(BasicModelConfig, self).__init__(**kwargs)
        self.embedding, self.embedding_config = self.get_embedding(kwargs.pop("embedding", "none"))
        self.encoder, self.encoder_config = self.get_encoder(kwargs.pop("encoder", "none"))
        self.decoder, self.decoder_config = self.get_decoder(kwargs.pop("decoder", "none"))
        self.config = kwargs.pop('config', {})

    def get_embedding(self, config):
        """get embedding config and embedding module

        :config: TODO
        :returns: TODO

        """
        return self._get_leaf_module(EMBEDDING_REGISTRY, EMBEDDING_CONFIG_REGISTRY, "embedding", config)
        
    def get_encoder(self, config):
        """get encoder config and encoder module

        :config: TODO
        :returns: TODO

        """
        return self._get_leaf_module(ENCODER_REGISTRY, ENCODER_CONFIG_REGISTRY, "encoder", config)
        
    def get_decoder(self, config):
        """get decoder config and decoder module

        :config: TODO
        :returns: TODO

        """
        return self._get_leaf_module(DECODER_REGISTRY, DECODER_CONFIG_REGISTRY, "decoder", config)


@model_register('basic')
class BasicModel(nn.Module):
    """
    Sequence labeling model
    """

    def __init__(self, config: BasicModelConfig):
        super().__init__()
        self.embedding = config.embedding(config.embedding_config)
        self.encoder = config.encoder(config.encoder_config)
        self.decoder = config.decoder(config.decoder_config)
        self.config = config.config

    def forward(self, **inputs: Dict)->Dict:
        """predict forward
        :**inputs: Dict: TODO
        :returns: TODO
        """
        embedding = self.embedding(**inputs)
        encode_outputs = self.encoder(embedding=embedding, **inputs)
        decode_outputs = self.decoder(encode_outputs, **inputs)
        return decode_outputs

    def training_step(self, **inputs: Dict)->Dict:
        """TODO: Docstring for training_step.

        :**inputs: Dict: TODO
        :returns: TODO
        """
        embedding = self.embedding.training_step(**inputs)
        encode_outputs = self.encoder.training_step(embedding=embedding, **inputs)
        decode_outputs = self.decoder.training_step(encode_outputs, **inputs)
        return decode_outputs

    def valid_step(self, **inputs: Dict)->Dict:
        """TODO: Docstring for training_step.

        :**inputs: Dict: TODO
        :returns: TODO

        """
        embedding = self.embedding.valid_step(**inputs)
        encode_outputs = self.encoder.valid_step(embedding=embedding, **inputs)
        decode_outputs = self.decoder.valid_step(encode_outputs, **inputs)
        return decode_outputs
