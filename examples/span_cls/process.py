# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import uuid

import pandas as pd
from datasets import load_dataset
from utils import convert

from dlk.preprocess import PreProcessor

# this is just for prepro the data, not the real label<->id pair in process.
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

# data = load_dataset(path="conll2003")
data = load_dataset(path="./tmp")

data = data.map(
    lambda one: {
        "tokens": one["tokens"],
        "ner_tags": [label_map[i] for i in one["ner_tags"]],
    }
)


filed_name_map = {"train": "train", "validation": "valid", "test": "test"}
json_data_map = {}
for filed in ["train", "test"]:
    filed_data = data[filed].to_dict()
    tokens = filed_data["tokens"]
    labels = filed_data["ner_tags"]
    inses = []
    for token, label in zip(tokens, labels):
        inses.append([token, label])
    json_data_map[filed_name_map[filed]] = convert(inses)


input = {
    "train": pd.DataFrame(json_data_map["train"]),
    "valid": pd.DataFrame(json_data_map["test"]),
}


processor = PreProcessor("./config/processor.jsonc")
processor.fit(input)
