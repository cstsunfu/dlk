from dataclasses import dataclass, field
from typing import List


@dataclass
class EncoderConfig:
    enc_layers: int = 2
    kernel_sizes: List[int] = field(default_factory=lambda: [3])


@dataclass
class ModelConfig:
    # --- Attention Mechanism Config ---
    num_attention_heads: int = (
        6  # Number of heads for MHA. hidden_size must be divisible by this.
    )
    use_rope: bool = True  # Whether to use Rotary Positional Embeddings (RoPE).
    max_position_embeddings: int = 512  # Max sequence length for RoPE.

    # --- Architectural choices ---
    # `alignment` is now replaced by the MHA module, so this string is no longer used.
    fusion: str = "full"  # Options: 'simple', 'full'
    connection: str = "aug"  # Options: 'none', 'residual', 'aug'
    prediction: str = "full"  # Options: 'simple', 'full', 'symmetric'

    # --- Model dimensions and parameters ---
    embedding_dim: int = 300
    hidden_size: int = 150  # Should be divisible by num_attention_heads
    dropout: float = 0.2
    blocks: int = 2
    fix_embeddings: bool = True
    num_vocab: int = 50000
    num_classes: int = 2

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
