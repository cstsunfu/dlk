# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
from src.get_data import get_data

from dlk.preprocess import PreProcessor

data = get_data()
input = {
    "train": pd.DataFrame(data).head(100),
    "valid": pd.DataFrame(data).head(100),
}

processor = PreProcessor("./config/processor.jsonc")
processor.fit(input)
