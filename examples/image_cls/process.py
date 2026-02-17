# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os
import uuid

from datasets import Dataset, load_dataset
from src.label import label_map

from dlk.preprocess import PreProcessor

# Ensure proxy if needed
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


def preprocess_images(batch):
    # Batch process: map label IDs to strings and generate UUIDs
    # HF Datasets handles 'image' column automatically as PIL images
    return {
        "image": batch["image"],
        "labels": [label_map[l] for l in batch["label"]],
        "uuid": [str(uuid.uuid4()) for _ in range(len(batch["label"]))],
    }


if __name__ == "__main__":
    # Load ImageNet (or subset)
    # data = load_dataset("imagenet-1k", split="train", streaming=True) # If large
    data = load_dataset("./data/imagenet")  # Local loading

    # Transform
    processed = data.map(
        preprocess_images,
        batched=True,
        remove_columns=["label"],
        writer_batch_size=1000,  # Efficient writing
    )

    # Split manually for demo purposes (HF Dataset slicing)
    input_data = {
        "train": processed["train"].select(range(10000)),
        "valid": processed["train"].select(range(10000, 10100)),
    }

    processor = PreProcessor("./config/processor.jsonc")
    processor.fit(input_data)
