# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import uuid

import pandas as pd
from datasets import load_dataset

from dlk.preprocess import PreProcessor


def flat(data):
    """flat the data like zip"""
    sentences = data["sentence"]
    uuids = data["uuid"]
    labelses = data["labels"]
    return [
        {"sentence": sentece, "labels": [labels], "uuid": uuid}
        for sentece, labels, uuid in zip(sentences, labelses, uuids)
    ]


label_map = {0: "neg", 1: "pos"}

data = load_dataset("sst2")
data = data.map(
    lambda one: {
        "sentence": one["sentence"],
        "labels": label_map[one["label"]],
        "uuid": str(uuid.uuid1()),
    },
    remove_columns=["sentence", "label"],
)
input = {
    "train": pd.DataFrame(flat(data["train"].to_dict())),
    "valid": pd.DataFrame(flat(data["validation"].to_dict())),
}

processor = PreProcessor("./config/processor.jsonc")
processor.fit(input)
