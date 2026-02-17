import math
from typing import Collection, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dlk.nn.model.re2.config import ModelConfig
from dlk.nn.utils.rope import RoFormerSinusoidalPositionalEmbedding


class Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fix_embeddings = args.fix_embeddings
        self.embedding = nn.Embedding(args.num_vocab, args.embedding_dim, padding_idx=0)
        self.dropout = args.dropout

    def set_(self, value):
        self.embedding.weight.requires_grad = not self.fix_embeddings
        self.embedding.load_state_dict({"weight": torch.tensor(value)})

    def forward(self, x):
        x = self.embedding(x)
        x = f.dropout(x, self.dropout, self.training)
        return x


class Linear(nn.Module):
    """
    A Linear layer with optional weight normalization, custom initialization, and GELU activation.
    This is kept from the original as it contains specific logic.
    """

    def __init__(self, in_features: int, out_features: int, activations: bool = False):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        # Custom initialization
        nn.init.normal_(
            linear.weight, std=math.sqrt((2.0 if activations else 1.0) / in_features)
        )
        nn.init.zeros_(linear.bias)

        modules = [nn.utils.weight_norm(linear)]
        if activations:
            modules.append(nn.GELU())  # Use standard GELU
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Conv1d(nn.Module):
    """
    A block of 1D convolutions with multiple kernel sizes.
    This is kept as it's a core component of the encoder's architecture.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_sizes: Collection[int]
    ):
        super().__init__()
        assert all(
            k % 2 == 1 for k in kernel_sizes
        ), "Only odd kernel sizes are supported"
        assert (
            out_channels % len(kernel_sizes) == 0
        ), "out_channels must be divisible by the number of kernels"

        # Each kernel produces a part of the output channels
        sub_out_channels = out_channels // len(kernel_sizes)

        convs = []
        for kernel_size in kernel_sizes:
            conv = nn.Conv1d(
                in_channels,
                sub_out_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
            )
            # Custom initialization
            nn.init.normal_(
                conv.weight, std=math.sqrt(2.0 / (in_channels * kernel_size))
            )
            nn.init.zeros_(conv.bias)
            convs.append(nn.Sequential(nn.utils.weight_norm(conv), nn.GELU()))

        self.model = nn.ModuleList(convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply each convolution and concatenate the results along the channel dimension
        return torch.cat([encoder(x) for encoder in self.model], dim=1)


class SimpleFusion(nn.Module):
    def __init__(self, config: ModelConfig, input_size: int):
        super().__init__()
        self.fusion = Linear(input_size * 2, config.hidden_size, activations=True)

    def forward(self, x: torch.Tensor, align: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([x, align], dim=-1))


class FullFusion(nn.Module):
    def __init__(self, config: ModelConfig, input_size: int):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.fusion1 = Linear(input_size * 2, config.hidden_size, activations=True)
        self.fusion2 = Linear(input_size * 2, config.hidden_size, activations=True)
        self.fusion3 = Linear(input_size * 2, config.hidden_size, activations=True)
        self.gate = Linear(config.hidden_size * 3, config.hidden_size, activations=True)

    def forward(self, x: torch.Tensor, align: torch.Tensor) -> torch.Tensor:
        x1 = self.fusion1(torch.cat([x, align], dim=-1))
        x2 = self.fusion2(torch.cat([x, x - align], dim=-1))
        x3 = self.fusion3(torch.cat([x, x * align], dim=-1))

        x_combined = torch.cat([x1, x2, x3], dim=-1)
        x_combined = self.dropout(x_combined)

        return self.gate(x_combined)


class NullConnection(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

    def forward(
        self, x: torch.Tensor, res: torch.Tensor, block_idx: int
    ) -> torch.Tensor:
        return x


class ResidualConnection(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear = Linear(config.embedding_dim, config.hidden_size)

    def forward(
        self, x: torch.Tensor, res: torch.Tensor, block_idx: int
    ) -> torch.Tensor:
        if block_idx == 1:
            res = self.linear(res)
        return (x + res) * math.sqrt(0.5)


class AugmentedConnection(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

    def forward(
        self, x: torch.Tensor, res: torch.Tensor, block_idx: int
    ) -> torch.Tensor:
        if block_idx == 1:
            # On the first block after embedding, 'res' is the original embedding
            return torch.cat([x, res], dim=-1)

        # On subsequent blocks, 'res' is the augmented feature from the previous block
        hidden_size = x.size(-1)
        # Add residual to the first part of the features
        x_residual = (res[:, :, :hidden_size] + x) * math.sqrt(0.5)
        # Concatenate with the augmented part (original embedding)
        return torch.cat([x_residual, res[:, :, hidden_size:]], dim=-1)


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig, input_size: int):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)

        layers = []
        current_size = input_size
        for _ in range(config.encoder.enc_layers):
            layers.append(
                Conv1d(
                    in_channels=current_size,
                    out_channels=config.hidden_size,
                    kernel_sizes=config.encoder.kernel_sizes,
                )
            )
            current_size = config.hidden_size

        self.encoders = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C), mask: (B, L)
        x = x.transpose(1, 2)  # -> (B, C, L)
        mask_conv = mask.unsqueeze(1)  # -> (B, 1, L)

        for i, encoder_layer in enumerate(self.encoders):
            x.masked_fill_(~mask_conv, 0.0)
            if i > 0:
                x = self.dropout(x)
            x = encoder_layer(x)

        x = self.dropout(x)
        return x.transpose(1, 2)  # -> (B, L, C)


class MaxPooling(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: (B, L), x: (B, L, C)
        mask_expanded = mask.unsqueeze(-1)  # -> (B, L, 1)
        x_masked = x.masked_fill(~mask_expanded, -float("inf"))
        return x_masked.max(dim=1)[0]


class SimplePrediction(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        input_features = config.hidden_size * 2
        self.dense = nn.Sequential(
            nn.Dropout(config.dropout),
            Linear(input_features, config.hidden_size, activations=True),
            nn.Dropout(config.dropout),
            Linear(config.hidden_size, config.num_classes),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.dense(torch.cat([a, b], dim=-1))


class FullPrediction(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        input_features = config.hidden_size * 4
        self.dense = nn.Sequential(
            nn.Dropout(config.dropout),
            Linear(input_features, config.hidden_size, activations=True),
            nn.Dropout(config.dropout),
            Linear(config.hidden_size, config.num_classes),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.dense(torch.cat([a, b, a - b, a * b], dim=-1))


class SymmetricPrediction(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        input_features = config.hidden_size * 4
        self.dense = nn.Sequential(
            nn.Dropout(config.dropout),
            Linear(input_features, config.hidden_size, activations=True),
            nn.Dropout(config.dropout),
            Linear(config.hidden_size, config.num_classes),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.dense(torch.cat([a, b, (a - b).abs(), a * b], dim=-1))


class MultiHeadAttention(nn.Module):
    """
    A custom Multi-Head Attention module that supports optional RoPE.
    It performs one attention operation: Q, K, V -> Output.
    """

    def __init__(self, config: ModelConfig, input_size: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.embed_dim = input_size
        self.use_rope = config.use_rope

        assert (
            self.embed_dim % self.num_heads == 0
        ), "Embedding dimension must be divisible by number of attention heads."

        self.head_dim = self.embed_dim // self.num_heads

        # Linear projections
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout = nn.Dropout(config.dropout)

        self.rope_emb = None
        if self.use_rope:
            # RoPE is applied to each head, so the dimension is head_dim
            self.rope_emb = RoFormerSinusoidalPositionalEmbedding(
                num_positions=config.max_position_embeddings,
                embedding_dim=self.head_dim,
            )

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, dim) -> (batch, num_heads, seq_len, head_dim)"""
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _combine_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """(batch, num_heads, seq_len, head_dim) -> (batch, seq_len, dim)"""
        batch_size, _, seq_len, _ = tensor.shape
        return (
            tensor.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):

        # 1. Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. Split into heads
        q = self._split_heads(q)  # (batch, num_heads, seq_len_q, head_dim)
        k = self._split_heads(k)  # (batch, num_heads, seq_len_k, head_dim)
        v = self._split_heads(v)  # (batch, num_heads, seq_len_k, head_dim)

        # 3. Apply RoPE if enabled
        if self.use_rope and self.rope_emb is not None:
            pos_emb = self.rope_emb(q.shape[2])
            q, k = (
                RoFormerSinusoidalPositionalEmbedding.apply_rotary_position_embeddings(
                    pos_emb, q, k
                )
            )

        # 4. Scaled Dot-Product Attention
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Apply padding mask
        if key_padding_mask is not None:
            # key_padding_mask is (batch, seq_len_k), needs to be broadcastable
            mask = key_padding_mask.unsqueeze(1).unsqueeze(
                2
            )  # -> (batch, 1, 1, seq_len_k)
            attn_scores = attn_scores.masked_fill(mask, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)

        # 5. Combine heads and final projection
        context = self._combine_heads(context)
        output = self.out_proj(context)

        return output
