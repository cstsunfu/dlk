# import pickle as pkl

# train = pkl.load(open('./test_cls/meta.pkl', 'rb'))
# print(train)

# data = pkl.load(open('./test_cls/processed_data.pkl', 'rb'))
# print(data)




import pandas as pd
from dlkit.utils.logger import setting_logger
setting_logger('process.log')
from dlkit.utils.parser import config_parser_register
import json
import hjson
from dlkit.data.processors import processor_config_register, processor_register

train = pd.DataFrame({
    "sentence": ['i love you.'.split()+['f fas f'], 'thank you'.split()],
    "label": ['pos', 'neg']
})

dev = pd.DataFrame({
    "sentence": ['thank you'.split()],
    "label": ['neg']
})

data = {"train": train, "dev": dev}

inp = {"data": data}

config = config_parser_register.get("processor")(hjson.load(open("./test_processor.hjson"),object_pairs_hook=dict)).parser_with_check()[0]
# print(json.dumps(config, indent=4))
# processor_config_register.get(config.get('_name'))(stage="train", config=config)
processor_register.get(config.get('_name'))(stage="train", config=processor_config_register.get(config.get('_name'))(stage="train", config=config)).process(inp)
