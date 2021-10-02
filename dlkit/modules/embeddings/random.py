import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
from dlkit.utils.config import Config
from dlkit.modules.embeddings import embedding_register, embedding_config_register, EmbeddingInput, EmbeddingOutput 

@embedding_config_register("random")
class EmbeddingConfig(Config):
    """docstring for EmbeddingConfig"""
    def __init__(self, **args):
        super(EmbeddingConfig, self).__init__()
        self.num_embeddings = args.get('vocab_size', 0)
        self.embedding_dim = args.get('embedding_dim', 128)
        self.padding_idx = args.get('padding_idx', 0)
        

@embedding_register("random")
class Embedding(nn.Module):
    def __init__(self, config: EmbeddingConfig):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=config.num_embeddings, embedding_dim=config.embedding_dim, padding_idx=config.padding_idx)

    def forward(self, inputs: EmbeddingInput)->EmbeddingOutput:
        """
        """
        # No padding necessary.
        seq_rep = self.embedding(inputs.input_ids)
        return EmbeddingOutput(represent=seq_rep)


if __name__ == "__main__":
    pass
