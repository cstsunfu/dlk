# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import uuid

from datasets import Dataset, load_dataset
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


filed_name_map = {"train": "train", "test": "test"}
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
    "train": Dataset.from_list(json_data_map["train"][:1000]),
    "valid": Dataset.from_list(json_data_map["test"][:100]),
}

processor = PreProcessor("./config/bert_firstpiece_lstm_crf/processor.jsonc")
processor.fit(input)
