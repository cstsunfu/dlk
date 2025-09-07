# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from datasets import Dataset
from src.get_data import get_data

from dlk.preprocess import PreProcessor

data: list = get_data()
input = {
    "train": Dataset.from_list(data),
    "valid": Dataset.from_list(data),
}

processor = PreProcessor("./config/processor.jsonc")
processor.fit(input)
