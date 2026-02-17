# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List

from intc import NestField, StrField, SubModule, cregister

from dlk.data.subprocessor import BaseSubProcessor, BaseSubProcessorConfig
from dlk.utils.register import register, register_module_name

logger = logging.getLogger(__name__)


@cregister("subprocessor", "teacher_label_gather")
class TeacherLabelGatherConfig(BaseSubProcessorConfig):
    """Configuration for gathering teacher labels offline."""

    submodule = SubModule(
        value={},
        suggestions=["teacher_fetcher"],
        help="Submodule for fetching the teacher's prediction.",
    )


@register("subprocessor", "teacher_label_gather")
class TeacherLabelGather(BaseSubProcessor):
    """SubProcessor to fetch teacher predictions and save them into the dataset."""

    def __init__(self, stage: str, config: TeacherLabelGatherConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.config = config

        # Initialize the fetcher via SubModule
        fetcher_configs = self.config._get_modules("teacher_fetcher")
        assert (
            len(fetcher_configs) == 1
        ), "Must provide exactly one teacher_fetcher config."

        fetcher_config = fetcher_configs[0]
        fetcher_name = register_module_name(fetcher_config._module_name)
        self.fetcher = register.get("teacher_fetcher", fetcher_name)(fetcher_config)

    def process_batch(
        self, batch: Dict[str, List[Any]], deliver_meta: bool = False
    ) -> Dict[str, List[Any]]:
        """Processes a batch to add teacher logits.

        Args:
            batch: The input batch dictionary.
            deliver_meta: Whether to deliver meta (unused here).

        Returns:
            The batch dictionary updated with teacher logits.
        """
        # Formulate requests dynamically based on input_map
        # e.g., input_map = {"text": "sentence", "uuid": "uuid"}
        # => extracts 'sentence' and 'uuid' from batch and formats them as 'text' and 'uuid' for the request
        batch_size = len(next(iter(batch.values())))
        request_data = [{} for _ in range(batch_size)]

        for req_key, batch_key in self.config.input_map.items():
            if batch_key in batch:
                for i in range(batch_size):
                    request_data[i][req_key] = batch[batch_key][i]
            else:
                logger.warning(
                    f"Key '{batch_key}' not found in batch, skipping for request mapping."
                )

        # Fetch logits from remote teacher(s)
        # Expected return: List of results matching batch size
        teacher_results = self.fetcher.fetch(request_data)

        # Save to batch dynamically based on output_map
        # e.g., output_map = {"teacher_logits": "teacher_logits"}
        for fetcher_key, batch_key in self.config.output_map.items():
            # Assuming fetcher returns directly the mapped item or we map it
            # For simplicity, if we fetch just one core output (logits), we assign it directly.
            # If the fetcher returned complex dicts, we could extract `fetcher_key`.
            batch[batch_key] = teacher_results

        return batch
