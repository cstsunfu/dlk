# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import uuid

from datasets import Dataset, load_dataset

from dlk.preprocess import PreProcessor


def preprocess_function(examples):
    # Create UUIDs and map labels
    label_map = {0: "neg", 1: "pos"}
    num_ex = len(examples["sentence"])
    return {
        "sentence": examples["sentence"],
        "labels": [label_map[l] for l in examples["label"]],
        "uuid": [str(uuid.uuid4()) for _ in range(num_ex)],
    }


if __name__ == "__main__":
    # Load from HF Hub
    data = load_dataset("sst2")
    data = data.filter(lambda x: x["label"] != -1)

    # Apply transformations using HF map
    data = data.map(preprocess_function, batched=True, remove_columns=["label", "idx"])

    # Input is now a dict of HF Datasets
    input_data = {
        "train": data["train"].select(range(100)),
        "valid": data["validation"].select(range(100)),
    }

    # The config must now use "dataset" as data_type
    processor = PreProcessor("./config/processor.jsonc")
    processor.fit(input_data)
