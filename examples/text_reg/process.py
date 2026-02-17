# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.
import uuid

from datasets import Dataset, load_dataset

from dlk.preprocess import PreProcessor


def preprocess_function(batch):
    """Process batch: label -> values (float list)."""
    return {
        "sentence": batch["sentence"],
        "values": [[float(l)] for l in batch["label"]],
        "uuid": [str(uuid.uuid4()) for _ in range(len(batch["label"]))],
    }


if __name__ == "__main__":
    data = load_dataset("sst2")

    data = data.map(preprocess_function, batched=True, remove_columns=["label", "idx"])

    input_data = {
        "train": data["train"],
        "valid": data["validation"],
    }

    processor = PreProcessor("./config/processor.jsonc")
    processor.fit(input_data)
