import pytorch_lightning as pl
from . import model_register, model_config_register, ModelConfig
from modules.embeddings import embedding_register, embedding_config_register


@model_config_register('seq_label')
class SeqLabelConfig(ModelConfig):
    """docstring for SeqLabelConfig"""
    def __init__(self, **kwargs):
        super(SeqLabelConfig, self).__init__(**kwargs)
        self.embedding, self.embedding_config = self.get_embedding(kwargs.pop("embedding", "word2vec"))
        self.encoder, self.encoder_config = self.get_encoder(kwargs.pop("encoder", "lstm"))
        self.decoder, self.decoder_config = self.get_decoder(kwargs.pop("decoder", "linear"))
        

@model_register('seq_label')
class SeqLabelModel(pl.LightningModule):
    """
    Sequence labeling model
    """

    def __init__(self, config: SeqLabelConfig):
        super().__init__()
        self.config = config
        self.embedding = config.embedding(config.embedding_config)
        self.encoder = config.embedding(config.encoder_config)
        self.decoder = config.embedding(config.decoder_config)

    def _check_input(self, **inputs):
        """TODO: Docstring for _check_input.
        :returns: TODO

        """
        return True


    def forward(self, **inputs):
        self._check_input(**inputs)
        input_ids = inputs.get('input_ids')
        input_mask = inputs.get('input_mask')
        embedding = self.embedding(input_ids)
        encoded_vector = self.encoder(embedding=embedding, input_mask=input_mask)['encoded_vector']
        decoded_vector = self.decoder(input=encoded_vector, input_mask)['decoded_vector']
        return decoded_vector

    def training_step(self):
        pass

    def validation_step(self):
        pass

    def test_step(self):
        pass

    def predict_step(self):
        pass

    def configure_optimizers(self):
        raise NotImplementedError
