# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import uuid

import src
from datasets import Dataset, load_dataset

from dlk.preprocess import PreProcessor


def preprocess_function(examples):
    label_map = {0: "neg", 1: "pos"}
    num_rows = len(examples["sentence"])
    return {
        "sentence": examples["sentence"],
        "labels": [label_map[l] for l in examples["label"]],
        "uuid": [str(uuid.uuid4()) for _ in range(num_rows)],
    }


if __name__ == "__main__":
    # Load
    data = load_dataset("sst2")

    # drop label == -1

    data = data.filter(lambda x: x["label"] != -1)

    # Map
    data = data.map(preprocess_function, batched=True, remove_columns=["label", "idx"])

    input_data = {"train": data["train"], "valid": data["validation"]}

    # The config must rely on "dataset" type
    processor = PreProcessor("./config/processor.jsonc")
    processor.fit(input_data)
