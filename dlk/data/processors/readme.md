## The subprocessor config format
In subprocessors, the config is based on the progress stage(train, predict, online, etc.). 

The stage config could be a dict, a str, or a tuple, for different type of config, we will parser the configure the different way. 
    1. when the config is a dict, this is the default type, all things go as you think.
    2. when the config is a str, the string must be one of stage name(train, predict, online, etc.) and the stage config is already defined as dict description in "1"
    3. when the config is a tuple(two elements list), the first element must be a str, which defined in "2", and the second element is a update config, which type is dict(or None) and defined in '1'

Some config value set to "*@*", this means you must provided this key-value pair in your own config

## Processor Config Example
```hjson
{
    "processor": {
        "_name": "test_text_classification",
        "config": {
            "feed_order": ["load", "tokenizer", "token_gather", "label_to_id", "save"]
        },
        "subprocessor@load": {
            "_name": "load",
            "config":{
                "base_dir": "."
                "predict":{
                    "token_ids": "./token_ids.pkl",
                    "embedding": "./embedding.pkl",
                    "label_ids": "./label_ids.pkl",
                },
                "online": [
                    "predict", //base predict
                    {   // special config, update predict, is this case, the config is null, means use all config from "predict"
                    }
                ]
            }
        },
        "subprocessor@save": {
            "_name": "save",
            "config":{
                "base_dir": "."
                "train":{
                    "data.train": "./train.pkl",
                    "data.dev": "./dev.pkl",
                    "token_ids": "./token_ids.pkl",
                    "embedding": "./embedding.pkl",
                    "label_ids": "./label_ids.pkl",
                },
                "predict": {
                    "data.predict": "./predict.pkl"
                }
            }
        },
        "subprocessor@tokenizer":{
            "_base": "wordpiece_tokenizer",
            "config": {   TODO: REfactor config
                "train": { // you can add some whitespace surround the '&' 
                    "data_set": {                   // for different stage, this processor will process different part of data
                        "train": ["train", "dev"],
                        "predict": ["predict"],
                        "online": ["online"]
                    },
                    "config_path": "./token.json",
                    "normalizer": ["nfd", "lowercase", "strip_accents", "some_processor_need_config": {config}], // if don't set this, will use the default normalizer from config
                    "pre_tokenizer": ["whitespace": {}], // if don't set this, will use the default normalizer from config
                    "post_processor": "bert", // if don't set this, will use the default normalizer from config, WARNING: not support disable  the default setting( so the default tokenizer.post_tokenizer should be null and only setting in this configure)
                    "filed_map": { // this is the default value, you can provide other name
                        "tokens": "tokens",
                        "ids": "ids",
                        "attention_mask": "attention_mask",
                        "type_ids": "type_ids",
                        "special_tokens_mask": "special_tokens_mask",
                        "offsets": "offsets",
                    }, // the tokenizer output(the key) map to the value
                    "data_type": "single", // single or pair, if not provide, will calc by len(process_data)
                    "process_data": [
                        ["sentence", { "is_pretokenized": false}], 
                    ],
                    /*"data_type": "pair", // single or pair*/
                    /*"process_data": [*/
                        /*['sentence_a', { "is_pretokenized": false}], */ 
                        /*['sentence_b', {}], the config of the second data must as same as the first*/ 
                    /*],*/
                },
                "predict": "train",
                "online": "train"
            }
        },
        "subprocessor@token_gather":{
            "_name": "token_gather",
            "config": {
                "train": { // only train stage using
                    "data_set": {                   // for different stage, this processor will process different part of data
                        "train": ["train", "dev"]
                    },
                    "gather_columns": ["label"], //List of columns. Every cell must be sigle token or list of tokens or set of tokens
                    "deliver": "label_vocab", // output Vocabulary object (the Vocabulary of labels) name. 
                    "update": null, // null or another Vocabulary object to update
                }
            }
        },
        "subprocessor@label_to_id":{
            "_name": "token2id",
            "config": {
                "train":{ //train、predict、online stage config,  using '&' split all stages
                    "data_pair": {
                        "label": "label_id"
                    },
                    "data_set": {                   // for different stage, this processor will process different part of data
                        "train": ['train', 'dev'],
                        "predict": ['predict'],
                        "online": ['online']
                    },
                    "vocab": "label_vocab", // usually provided by the "token_gather" module
                },
                "predict": "train",
                "online": "train",
            }
        }
    }
}

```



## To Process Data Format Example

You can provide dataframe format by yourself, or use the task_name_loader(if provided or you can write one) to load your dict format data to dataframe

```hjson
{
    "data": {
        "train": pd.DataFrame, // may include these columns "uuid"、"origin"、"label"
        "dev": pd.DataFrame, // may include these columns "uuid"、"origin"、"label"
    }
}

```

## Processed Data Format Example

```hjson
{
    "data": {
        "train": pd.DataFrame, // may include these columns "uuid"、"origin"、"labels"、"origin_tokens"、"label_ids"、"origin_token_ids"
        "dev": pd.DataFrame, // may include these columns "uuid"、"origin"、"labels"、"origin_tokens"、"label_ids"、"origin_token_ids"
    },
    "embedding": ..,
    "token_vocab": ..,
    "label_vocab": ..,
    ...
}
```
