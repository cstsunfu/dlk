import torch.nn as nn
from . import model_register, model_config_register, ModelConfig
from typing import Dict
# from modules.embeddings import embedding_register, embedding_config_register


@model_config_register('basic')
class BasicConfig(ModelConfig):
    """docstring for BasicConfig"""
    def __init__(self, **kwargs):
        super(BasicConfig, self).__init__(**kwargs)
        self.embedding, self.embedding_config = self.get_embedding(kwargs.pop("embedding", "none"))
        self.encoder, self.encoder_config = self.get_encoder(kwargs.pop("encoder", "none"))
        self.decoder, self.decoder_config = self.get_decoder(kwargs.pop("decoder", "none"))
        self.config = kwargs.pop('config', {})
        

@model_register('basic')
class BasicModel(nn.Module):
    """
    Sequence labeling model
    """

    def __init__(self, config: BasicConfig):
        super().__init__()
        self.embedding = config.embedding(config.embedding_config)
        self.encoder = config.encoder(config.encoder_config)
        self.decoder = config.decoder(config.decoder_config)
        self.config = config.config

    def _check_input(self, **inputs):
        """TODO: Docstring for _check_input.
        :returns: TODO

        """
        return True


    def forward(self, **inputs: Dict)->Dict:
        self._check_input(**inputs)
        embedding = self.embedding(**inputs)
        encode_outputs = self.encoder(embedding=embedding, **inputs)
        decode_outputs = self.decoder(encode_outputs, **inputs)
        return decode_outputs
