# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List

import numpy as np
from intc import MISSING, Base, ListField, NestField, StrField, cregister
from transformers import ViTImageProcessor

from dlk.data.subprocessor import BaseSubProcessor, BaseSubProcessorConfig
from dlk.utils.register import register

logger = logging.getLogger(__name__)

hf_image_processor = {"vit": ViTImageProcessor}


@cregister("subprocessor", "image_process")
class ImageProcessConfig(BaseSubProcessorConfig):
    """the image process subprocessor"""

    train_data_set = ListField(
        value=["train", "valid", "test"],
        suggestions=[["train", "valid", "test"]],
        help="the data set should be processed for train stage",
    )
    predict_data_set = ListField(
        value=["predict"],
        suggestions=[["predict"]],
        help="the data set should be processed for predict stage",
    )
    online_data_set = ListField(
        value=["online"],
        suggestions=[["online"]],
        help="the data set should be processed for online stage",
    )

    preprocess_config = StrField(
        value=MISSING,
        suggestions=["preprocess_config.json"],
        help="the hf image preprocess config path",
    )
    preprocess_method = StrField(
        value="vit",
        options=list(hf_image_processor.keys()),
        help="the hf image preprocess method",
    )

    class InputMap:
        image = StrField(
            value="image",
            suggestions=["image"],
            help="the input image",
        )

    input_map = NestField(value=InputMap, help="the input map")

    class OutputMap:
        pixel_values = StrField(
            value="pixel_values",
            suggestions=["pixel_values"],
            help="the processed image values",
        )

    output_map = NestField(value=OutputMap, help="the output map")


@register("subprocessor", "image_process")
class ImageProcess(BaseSubProcessor):
    """Preprocess images using HuggingFace ImageProcessor."""

    def __init__(self, stage: str, config: ImageProcessConfig, meta_dir: str):
        super().__init__(stage, config, meta_dir)
        self.config = config

        self.image_processor = hf_image_processor[
            self.config.preprocess_method
        ].from_json_file(self.config.preprocess_config)

    def process_batch(
        self, batch: Dict[str, List[Any]], deliver_meta: bool = False
    ) -> Dict[str, List[Any]]:
        """Process a batch of images.

        Args:
            batch: Input batch containing images (PIL or path).
            deliver_meta: Unused.

        Returns:
            Batch with added pixel_values.
        """
        input_col = self.config.input_map.image
        output_col = self.config.output_map.pixel_values

        if input_col in batch:
            images = batch[input_col]
            # HF ImageProcessor accepts a list of images
            # return_tensors='np' fits well with HF datasets
            try:
                encodings = self.image_processor(images, return_tensors="np")
                batch[output_col] = list(encodings.pixel_values)
            except Exception as e:
                logger.error(f"Failed to process images: {e}")
                # Fallback or re-raise based on requirements.
                # For batch processing, we usually want to fail fast or return empty.
                raise e

        return batch
