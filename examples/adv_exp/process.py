# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.
import uuid

import pandas as pd
from datasets import Dataset, load_dataset

from dlk.preprocess import PreProcessor

# Label mapping
label_map = {0: "entails", 1: "nor", 2: "contradicts"}


def preprocess_function(batch):
    """Process batch of data: map labels, rename columns, add uuid."""
    return {
        "sentence_a": batch["hypothesis"],
        "sentence_b": batch["premise"],
        "labels": [label_map[l] for l in batch["label"]],
        "uuid": [str(uuid.uuid4()) for _ in range(len(batch["label"]))],
    }


if __name__ == "__main__":
    # Load raw dataset (assuming ./data/origin exists or use "snli" from hub)
    # Using "snli" from hub for reproduction consistency if local data is missing
    try:
        data = load_dataset("./data/origin")
    except:
        data = load_dataset("snli")

    # Filter invalid labels (-1)
    data = data.filter(lambda x: x["label"] in label_map)

    # Transform
    data = data.map(
        preprocess_function,
        batched=True,
        remove_columns=["hypothesis", "premise", "label"],
    )

    # Split/Select for demo
    input_data = {
        "train": data["train"].select(range(10000)),
        "valid": data["validation"].select(range(1000)),
    }

    processor = PreProcessor("./config/processor.jsonc")
    processor.fit(input_data)
