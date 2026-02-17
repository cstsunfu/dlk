import torch
import torch.nn as nn

from dlk.nn.model.re2.config import ModelConfig
from dlk.nn.model.re2.modules import (
    AugmentedConnection,
    Embedding,
    Encoder,
    FullFusion,
    FullPrediction,
    Linear,
    MaxPooling,
    MultiHeadAttention,
    NullConnection,
    ResidualConnection,
    SimpleFusion,
    SimplePrediction,
    SymmetricPrediction,
)

MODULE_REGISTRY = {
    "fusion": {"simple": SimpleFusion, "full": FullFusion},
    "connection": {
        "none": NullConnection,
        "residual": ResidualConnection,
        "aug": AugmentedConnection,
    },
    "prediction": {
        "simple": SimplePrediction,
        "full": FullPrediction,
        "symmetric": SymmetricPrediction,
    },
}


class ReS2T(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config)

        input_emb_size = config.embedding_dim if config.connection == "aug" else 0

        self.blocks = nn.ModuleList()
        for i in range(config.blocks):
            enc_input_size = (
                config.embedding_dim
                if i == 0
                else (input_emb_size + config.hidden_size)
            )
            # The input to attention/fusion is the concatenation of the block's input and encoder's output
            attention_fusion_input_size = enc_input_size + config.hidden_size

            fusion_cls = MODULE_REGISTRY["fusion"][config.fusion]

            block = nn.ModuleDict(
                {
                    "encoder": Encoder(config, enc_input_size),
                    # Directly instantiate MultiHeadAttention instead of looking up in a registry
                    "attention": MultiHeadAttention(
                        config, attention_fusion_input_size
                    ),
                    "fusion": fusion_cls(config, attention_fusion_input_size),
                }
            )
            self.blocks.append(block)

        connection_cls = MODULE_REGISTRY["connection"][config.connection]
        self.connection = connection_cls(config)
        self.pooling = MaxPooling()
        prediction_cls = MODULE_REGISTRY["prediction"][config.prediction]
        self.prediction = prediction_cls(config)

    def forward(self, inputs: dict) -> torch.Tensor:
        a, b = inputs["text1"], inputs["text2"]
        mask_a, mask_b = inputs["mask1"].bool(), inputs["mask2"].bool()

        a_emb = self.embedding(a)
        b_emb = self.embedding(b)
        res_a, res_b = a_emb, b_emb

        for i, block in enumerate(self.blocks):
            if i > 0:
                a_emb, b_emb = self.connection(a_emb, res_a, i), self.connection(
                    b_emb, res_b, i
                )
                res_a, res_b = a_emb, b_emb

            a_enc = block["encoder"](a_emb, mask_a)
            b_enc = block["encoder"](b_emb, mask_b)

            a_cat = torch.cat([a_emb, a_enc], dim=-1)
            b_cat = torch.cat([b_emb, b_enc], dim=-1)

            # --- Perform Cross-Attention ---
            # Create attention mask where True indicates padding.
            mask_a_pad = ~mask_a
            mask_b_pad = ~mask_b

            # To get features for sentence 'a', 'a' is the query, and it attends to 'b' (key, value)
            align_a = block["attention"](
                query=a_cat, key=b_cat, value=b_cat, key_padding_mask=mask_b_pad
            )

            # To get features for sentence 'b', 'b' is the query, and it attends to 'a' (key, value)
            align_b = block["attention"](
                query=b_cat, key=a_cat, value=a_cat, key_padding_mask=mask_a_pad
            )

            a_fused = block["fusion"](a_cat, align_a)
            b_fused = block["fusion"](b_cat, align_b)

            a_emb, b_emb = a_fused, b_fused

        a_pool, b_pool = self.pooling(a_emb, mask_a), self.pooling(b_emb, mask_b)
        return self.prediction(a_pool, b_pool)


class SentenceClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config)
        connection_cls = MODULE_REGISTRY["connection"][config.connection]
        self.connection = connection_cls(config)

        self.blocks = nn.ModuleList()
        for i in range(config.blocks):
            block = nn.ModuleDict()
            enc_input_size = (
                config.embedding_dim
                if i == 0
                else (
                    config.hidden_size
                    + (config.embedding_dim if config.connection == "aug" else 0)
                )
            )
            block["encoder"] = Encoder(config, enc_input_size)

            if self.config.num_attention_heads > 0:
                attention_fusion_input_size = enc_input_size + config.hidden_size
                fusion_cls = MODULE_REGISTRY["fusion"][config.fusion]
                block["attention"] = MultiHeadAttention(
                    config, attention_fusion_input_size
                )
                block["fusion"] = fusion_cls(config, attention_fusion_input_size)
            self.blocks.append(block)

        self.pooling = MaxPooling()
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            Linear(config.hidden_size, config.hidden_size, activations=True),
            nn.Dropout(config.dropout),
            Linear(config.hidden_size, config.num_classes),
        )

    def forward(self, inputs: dict) -> torch.Tensor:
        x_ids, mask = inputs["text"], inputs["mask"].bool()
        x_processed = self.embedding(x_ids)
        res_emb = x_processed

        for i, block in enumerate(self.blocks):
            if i > 0:
                x_processed = self.connection(x_processed, res_emb, i)
                if self.config.connection == "aug":
                    res_emb = x_processed

            x_enc = block["encoder"](x_processed, mask)

            if self.config.num_attention_heads > 0:
                x_cat = torch.cat([x_processed, x_enc], dim=-1)

                # --- Perform Self-Attention ---
                # Create padding mask (True for padding)
                padding_mask = ~mask
                aligned_x = block["attention"](
                    query=x_cat, key=x_cat, value=x_cat, key_padding_mask=padding_mask
                )
                x_processed = block["fusion"](x_cat, aligned_x)
            else:
                x_processed = x_enc

        pooled_output = self.pooling(x_processed, mask)
        return self.classifier(pooled_output)
