# import pickle as pkl

# train = pkl.load(open('./train.pkl', 'rb'))
# for i in train:
    # print(train[i])

# || 0    [i, love, youafsdafasd., f fas f]
# || 1                         [thank, you]
# || Name: sentence, dtype: object
# || 0    pos
# || 1    neg
# || Name: label, dtype: object
# || 0    [[CLS], i, love, you, ##af, ##sd, ##af, ##as, ...
# || 1                           [[CLS], thank, you, [SEP]]
# || Name: tokens, dtype: object
# || 0    [1, 151, 11157, 10855, 19292, 69246, 19292, 10...
# || 1                                 [1, 67715, 10855, 2]
# || Name: ids, dtype: object
# || 0    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# || 1                                  [1, 1, 1, 1]
# || Name: attention_mask, dtype: object
# || 0    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# || 1                                  [0, 0, 0, 0]
# || Name: type_ids, dtype: object
# || 0    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
# || 1                                  [1, 0, 0, 1]
# || Name: special_tokens_mask, dtype: object
# || 0    [(0, 0), (0, 1), (0, 4), (0, 3), (3, 5), (5, 7...
# || 1                     [(0, 0), (0, 5), (0, 3), (0, 0)]
# || Name: offsets, dtype: object
# || 0    [None, 0, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, None]
# || 1                                  [None, 0, 1, None]
# || Name: word_ids, dtype: object
# || 0    []
# || 1    []
# || Name: overflowing, dtype: object
# || 0    [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None]
# || 1                                  [None, 0, 0, None]
# || Name: sequence_ids, dtype: object
# || 0    0
# || 1    1
# || Name: label_id, dtype: int64
import pandas as pd
from dlkit.utils.parser import config_parser_register
import json
import hjson
from dlkit.processors import processor_config_register, processor_register

train = pd.DataFrame({
    "sentence": ['i love youafsdafasd.'.split()+['f fas f'], 'thank you'.split()],
    "label": ['pos', 'neg']
})

dev = pd.DataFrame({
    "sentence": ['thank you'.split()],
    "label": ['neg']
})

data = {"train": train, "dev": dev}

inp = {"data": data}

config = config_parser_register.get("processor")(hjson.load(open("./test_processor.hjson"),object_pairs_hook=dict)).parser_with_check()[0]
# # print(json.dumps(config, indent=4))
# processor_config_register.get(config.get('_name'))(stage="train", config=config)
processor_register.get(config.get('_name'))(stage="train", config=processor_config_register.get(config.get('_name'))(stage="train", config=config)).process(inp)



# {
    # "subprocessor@label_to_id": {
        # "config": {
            # "train": {
                # "data_pair": {
                    # "label": "label_id"
                # },
                # "data_set": {
                    # "train": [
                        # "train",
                        # "dev"
                    # ],
                    # "predict": [
                        # "predict"
                    # ],
                    # "online": [
                        # "online"
                    # ]
                # },
                # "vocab": "label_vocab"
            # },
            # "predict": "train",
            # "online": "train"
        # },
        # "_name": "token2id"
    # },
    # "subprocessor@token_gather": {
        # "config": {
            # "train": {
                # "data_set": {
                    # "train": [
                        # "train",
                        # "dev"
                    # ]
                # },
                # "gather_columns": [
                    # "label"
                # ],
                # "deliver": "label_vocab",
                # "update": null,
                # "unk": ""
            # }
        # },
        # "_name": "token_gather"
    # },
    # "subprocessor@tokenizer": {
        # "config": {
            # "train": {
                # "data_set": {
                    # "train": [
                        # "train",
                        # "dev"
                    # ],
                    # "predict": [
                        # "predict"
                    # ],
                    # "online": [
                        # "online"
                    # ]
                # },
                # "config_path": "./tokenizer.json",
                # "normalizer": "default",
                # "pre_tokenizer": "default",
                # "post_processor": "bert",
                # "filed_map": {
                    # "tokens": "tokens",
                    # "ids": "ids",
                    # "attention_mask": "attention_mask",
                    # "type_ids": "type_ids",
                    # "special_tokens_mask": "special_tokens_mask",
                    # "offsets": "offsets"
                # },
                # "data_type": "single",
                # "process_data": [
                    # [
                        # "sentence",
                        # {
                            # "is_pretokenized": false
                        # }
                    # ]
                # ]
            # },
            # "predict": "train",
            # "online": "train"
        # },
        # "_name": "wordpiece_tokenizer"
    # },
    # "subprocessor@save": {
        # "config": {
            # "base_dir": "./test_cls",
            # "train": {
                # "data.train": "./train.pkl",
                # "data.dev": "./dev.pkl",
                # "label_vocab": "./label_vocab.pkl"
            # },
            # "predict": {
                # "data.predict": "./predict.pkl"
            # }
        # },
        # "_name": "save"
    # },
    # "subprocessor@load": {
        # "config": {
            # "base_dir": "./test_cls",
            # "predict": {
                # "label_vocab": "./label_vocab.pkl"
            # },
            # "online": [
                # "predict",
                # {}
            # ]
        # },
        # "_name": "load"
    # },
    # "config": {
        # "feed_order": [
            # "load",
            # "tokenizer",
            # "token_gather",
            # "label_to_id",
            # "save"
        # ]
    # },
    # "_name": "basic@test_text_cls"
# }
