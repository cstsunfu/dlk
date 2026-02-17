# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List

import torch
from intc import SubModule, cregister

from dlk.imodel.default import DefaultIModel, DefaultIModelConfig
from dlk.utils.register import register, register_module_name

logger = logging.getLogger(__name__)


@cregister("imodel", "distill")
class DistillIModelConfig(DefaultIModelConfig):
    """IModel for online distillation. Fetches teacher labels dynamically."""

    submodule = SubModule(
        value={"teacher_fetcher": {"_base": "http"}},
        suggestions=[
            "model",
            "optimizer",
            "scheduler",
            "loss",
            "postprocessor",
            "adv_method",
            "teacher_fetcher",
        ],
    )


@register("imodel", "distill")
class DistillIModel(DefaultIModel):
    """IModel that injects teacher predictions dynamically during training."""

    def __init__(
        self, config: DistillIModelConfig, checkpoint=False, rt_config: Dict = {}
    ):
        super().__init__(config, checkpoint, rt_config)

        fetcher_configs = config._get_modules("teacher_fetcher")
        assert (
            len(fetcher_configs) == 1
        ), "Must provide exactly one teacher_fetcher config"
        fetcher_config = fetcher_configs[0]
        fetcher_name = register_module_name(fetcher_config._module_name)
        self.fetcher = register.get("teacher_fetcher", fetcher_name)(fetcher_config)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Overrides training_step to fetch teacher logits before forwarding.

        Args:
            batch: A mini-batch of inputs.
            batch_idx: The index of the mini-batch.

        Returns:
            The computed loss.
        """
        # Note: 'sentence' or raw text needs to be in the batch for HTTP API.
        # If your CollateFn removes it, you need to configure CollateFn to keep it
        # using `key_no_padding` or similar mechanisms.

        if "teacher_logits" not in batch:
            # Reconstruct dict list for the fetcher
            # Warning: Doing this inside training_step will block the GPU!
            batch_size = len(batch.get("input_ids", [0]))
            req_data = []

            # Assuming 'sentence' is passed through the dataloader for online fetching
            if "sentence" in batch:
                for i in range(batch_size):
                    req_data.append({"sentence": batch["sentence"][i]})

                teacher_logits = self.fetcher.fetch(req_data)
                batch["teacher_logits"] = torch.tensor(
                    teacher_logits, device=self.device
                )
            else:
                logger.warning(
                    "No 'sentence' found in batch, skipping online teacher fetching."
                )

        return super().training_step(batch, batch_idx)
