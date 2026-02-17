# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import uuid

from datasets import load_dataset

from dlk.preprocess import PreProcessor

label_map = {0: "entails", 1: "nor", 2: "contradicts"}


def preprocess_function(batch):
    return {
        "sentence_a": batch["hypothesis"],
        "sentence_b": batch["premise"],
        "labels": [label_map[l] for l in batch["label"]],
        "uuid": [str(uuid.uuid4()) for _ in range(len(batch["label"]))],
    }


if __name__ == "__main__":
    data = load_dataset("snli")
    data = data.filter(lambda x: x["label"] in label_map)
    data = data.map(
        preprocess_function,
        batched=True,
        remove_columns=["hypothesis", "premise", "label"],
    )

    input_data = {
        "train": data["train"].select(range(100)),
        "valid": data["test"].select(range(100)),
    }

    processor = PreProcessor("./config/processor.jsonc")
    processor.fit(input_data)
