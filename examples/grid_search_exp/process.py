# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import uuid

import pandas as pd
from datasets import load_dataset

from dlk.preprocess import PreProcessor


def flat(data):
    sentence_as = data["sentence_a"]
    sentence_bs = data["sentence_b"]
    uuids = data["uuid"]
    labelses = data["labels"]
    return [
        {
            "sentence_a": sentence_a,
            "sentence_b": sentence_b,
            "labels": [labels],
            "uuid": uuid,
        }
        for sentence_a, sentence_b, labels, uuid in zip(
            sentence_as, sentence_bs, labelses, uuids
        )
    ]


label_map = {0: "entails", 1: "nor", 2: "contradicts", -1: "remove"}

data = load_dataset("./data/origin")
data = data.map(
    lambda one: {
        "sentence_a": one["hypothesis"],
        "sentence_b": one["premise"],
        "labels": label_map[one["label"]],
        "uuid": str(uuid.uuid1()),
    },
    remove_columns=["hypothesis", "label", "premise"],
)
data = data.filter(lambda one: one["labels"] in {"entails", "nor", "contradicts"})

input = {
    "train": pd.DataFrame(flat(data["train"].to_dict())).head(10000),
    "valid": pd.DataFrame(flat(data["validation"].to_dict())).head(1000),
}

processor = PreProcessor("./config/processor.jsonc")
processor.fit(input)
