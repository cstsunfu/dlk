
import pandas as pd
from dlkit.utils.parser import config_parser_register
import json
import hjson

train = pd.DataFrame({
    "sentence": ['i love you.', 'thank you'],
    "label": ['pos', 'neg']
})

dev = pd.DataFrame({
    "sentence": ['thank you'],
    "label": ['neg']
})

data = {"train": train, "dev": dev}

inp = {"data": data}


print(json.dumps(config_parser_register.get("processor")(hjson.load(open("./test_processor.hjson"), object_pairs_hook=dict)).parser_with_check(), indent=4))

