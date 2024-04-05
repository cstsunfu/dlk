# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import copy
import io
import json
import os
import random
import uuid

import numpy as np
from PIL import Image
from src.label import label_map

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

import pandas as pd
from datasets import load_dataset

from dlk.preprocess import PreProcessor


def flat(data):
    """flat the data like zip"""
    images = data["image"]
    uuids = data["uuid"]
    labels = data["label"]
    # print(Image.open(io.BytesIO(images[0]["bytes"])).size)
    result = []
    for image, label, uuid in zip(images, labels, uuids):
        image = Image.open(io.BytesIO(image["bytes"]))
        if image.mode != "RGB":
            continue
        result.append(
            {
                "image": image,
                "labels": [label_map[label]],
                "uuid": uuid,
            }
        )
    random.shuffle(result)
    return result


# data = load_dataset("evanarlian/imagenet_1k_resized_256")
data = load_dataset("./data/imagenet")
data = data.map(
    lambda one: {
        "image": one["image"],
        "label": one["label"],
        "uuid": str(uuid.uuid1()),
    },
)
flat_train = flat(data["train"].to_dict())
# flat_val = flat(data['validation'].to_dict())
input = {
    "train": pd.DataFrame(flat_train).head(10000),
    "valid": pd.DataFrame(flat_train).head(100),
}

processor = PreProcessor("./config/processor.jsonc")
processor.fit(input)
