# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
import torch.nn.functional as F
from intc import MISSING, Base, BoolField, StrField, SubModule, cregister

from dlk.nn.base_module import BaseModel
from dlk.utils.register import register, register_module_name


@cregister("model", "two_tower")
class TwoTowerModelConfig(Base):
    """Configuration for Two Tower Contrastive Model."""

    share_encoder = BoolField(
        value=False,
        help="Whether Tower A and Tower B share the same encoder weights (e.g., for SimCSE).",
    )
    anchor_encoder_name = StrField(
        value="encoder#anchor",
        help="The specific register name of the anchor encoder in the submodule.",
    )
    positive_encoder_name = StrField(
        value="encoder#positive",
        help="The specific register name of the positive encoder in the submodule.",
    )
    submodule = SubModule(
        value={},
        suggestions=["encoder", "module"],
        help="Submodules defining the encoders and optional projection heads.",
    )


@register("model", "two_tower")
class TwoTowerModel(BaseModel):
    """Two-Tower Architecture for Contrastive Learning (CLIP, SimCSE, DPR).

    This model encodes two views (anchor and positive) into a shared embedding space
    and performs L2 normalization, making it extremely efficient for InfoNCE loss.
    """

    def __init__(self, config: TwoTowerModelConfig, checkpoint: bool = False):
        """Initializes the Two Tower model.

        Args:
            config (TwoTowerModelConfig): The model configuration.
            checkpoint (bool): Whether loading from a checkpoint.
        """
        super().__init__()
        self.config = config

        modules_dict = config._get_named_modules()

        # Initialize Tower A (Anchor)
        enc_a_config = modules_dict[f"@{config.anchor_encoder_name}"]
        self.encoder_a = register.get(
            "encoder", register_module_name(enc_a_config._module_name)
        )(enc_a_config)

        # Initialize Tower B (Positive)
        if config.share_encoder:
            self.encoder_b = self.encoder_a
        else:
            enc_b_config = modules_dict[f"@{config.positive_encoder_name}"]
            self.encoder_b = register.get(
                "encoder", register_module_name(enc_b_config._module_name)
            )(enc_b_config)

        # Optional Projection Head
        proj_configs = config._get_named_modules("module")
        self.projection_head = None
        if proj_configs:
            proj_name = list(proj_configs.keys())[0]
            self.projection_head = register.get(
                "module", register_module_name(proj_configs[proj_name]._module_name)
            )(proj_configs[proj_name])

        if not checkpoint:
            init_method_configs = config._get_modules("initmethod")
            init_method = register.get("initmethod", "default")()
            if init_method_configs:
                init_method = register.get(
                    "initmethod",
                    register_module_name(init_method_configs[0]._module_name),
                )(init_method_configs[0])
            self.encoder_a.init_weight(init_method)
            if not self.config.share_encoder:
                self.encoder_b.init_weight(init_method)
            if self.projection_head:
                self.projection_head.init_weight(init_method)

    def _extract_embedding(
        self, encoder, inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Extracts and normalizes the embedding from a given encoder."""
        enc_out = encoder(inputs)

        emb_key = encoder.config.output_map.embedding
        if emb_key not in enc_out:
            emb_key = list(enc_out.keys())[-1]  # Fallback to last output

        emb = enc_out[emb_key]
        if len(emb.shape) == 3:
            emb = emb[:, 0, :]  # CLS Pooling

        if self.projection_head:
            emb = self.projection_head(emb)
            if len(emb.shape) == 3:
                emb = emb[:, 0, :]

        return F.normalize(emb, p=2, dim=1)

    def training_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            inputs (Dict[str, torch.Tensor]): A mini-batch of inputs.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing normalized embeddings.
        """
        anchor_emb = self._extract_embedding(self.encoder_a, inputs)
        positive_emb = self._extract_embedding(self.encoder_b, inputs)

        outputs = {"anchor_emb": anchor_emb, "positive_emb": positive_emb}

        # Note: If explicit hard negatives are present, we could add logic here
        # to encode them using encoder_b and store in "negative_emb".
        return outputs

    def predict_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for prediction (generating single tower embeddings)."""
        # Determine which tower to use based on inputs provided.
        # The PostProcessor will use these to calculate recall/similarity.
        # By default, use anchor tower for inference.
        emb = self._extract_embedding(self.encoder_a, inputs)
        return {"embedding": emb, "_index": inputs["_index"]}
