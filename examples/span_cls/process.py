# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.


import pandas as pd
from datasets import Dataset, load_dataset
from utils import convert

from dlk.preprocess import PreProcessor

# NER Label Mapping
label_map = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}


def convert_to_span_format(batch):
    """
    Convert BIO format to Span format.
    Handles Integer -> String mapping AND structural conversion.
    """
    batch_inses = []

    for tokens, tags in zip(batch["tokens"], batch["ner_tags"]):

        labels = [label_map[i] for i in tags]

        ins = [tokens, labels]
        batch_inses.append(ins)

    converted_list = convert(batch_inses)

    if not converted_list:
        return {}

    output = {k: [] for k in converted_list[0].keys()}
    for item in converted_list:
        for k, v in item.items():
            output[k].append(v)

    return output


if __name__ == "__main__":

    # Load Data
    data = load_dataset("lhoestq/conll2003")  # HF Hub fallback

    # remove columns that are replaced/unused
    columns_to_remove = [
        c for c in data["train"].column_names if c not in ["tokens"]
    ]  # convert handles tokens -> sentence
    data = data.map(
        convert_to_span_format, batched=True, remove_columns=data["train"].column_names
    )

    input_data = {
        "train": data["train"],
        "valid": data["test"],  # Using test as valid for demo consistency
    }

    processor = PreProcessor("./config/processor.jsonc")
    processor.fit(input_data)
