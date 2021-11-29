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

train_data = json.load(open('../NER/train_format_label_data.json', 'r'))
valid_data = json.load(open('../NER/valid_format_label_data.json', 'r'))
# train = pd.DataFrame({
    # "ner_format_input": format_data
# })


# data = {"train": train}

# inp = {"data": data}


# train = pd.DataFrame({
    # "sentence": ['i love you.'.split()+['f fas f']+ [str(i%10)] for i in range(100)],
    # "labels": ['pos', 'neg']*50
# })

# valid = pd.DataFrame({
    # "sentence": ['Hi ou.'.split()+['f fas f']+ [str(i%10)] for i in range(100)],
    # "labels": ['pos', 'neg']*50
# })

data = {"train": train_data, "valid": valid_data}

inp = {"data": data}
# config = config_parser_register.get("processor")(hjson.load(open("./jobs/simple_cls/config.hjson"),object_pairs_hook=dict)).parser_with_check()[0]

config = config_parser_register.get("processor")(hjson.load(open("./jobs/simple_ner/config.hjson"),object_pairs_hook=dict)).parser_with_check()[0]
# print(json.dumps(config, indent=4))
processor_register.get(config.get('_name'))(stage="train", config=processor_config_register.get(config.get('_name'))(stage="train", config=config)).process(inp)
