# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from datasets import Dataset
from src.get_data import get_data

from dlk.preprocess import PreProcessor

if __name__ == "__main__":
    # get_data returns a list of dicts: [{'image': PIL.Image, 'target': str, 'uuid': str}, ...]
    data_list = get_data()

    # Create HF Dataset directly from list of dicts
    # HF Datasets automatically infers PIL.Image as Image feature
    dataset = Dataset.from_list(data_list)

    # Split (Manually for demo)
    input_data = {
        "train": dataset.select(range(min(100, len(dataset)))),
        "valid": dataset.select(range(min(100, len(dataset)))),
    }

    processor = PreProcessor("./config/processor.jsonc")
    processor.fit(input_data)
