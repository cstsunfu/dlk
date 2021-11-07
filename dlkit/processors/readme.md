%% TODO: Must be refactor process feed

In subprocessors, the config is based on the progress stage(train, predict, online, etc.). 

The stage config could be a dict, a str, or a tuple, for different type of config, we will parser the configure the different way. 
    1. when the config is a dict, this is the default type, all things go as you think.
    2. when the config is a str, the string must be one of stage name(train, predict, online, etc.) and the stage config is already defined as dict description in "1"
    3. when the config is a tuple(two elements list), the first element must be a str, which defined in "2", and the second element is a update config, which type is dict(or None) and defined in '1'

Some config value set to "*@*", this means you must provided this key-value pair in your own config

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
                }, //3
                "predict": "train",
                "online": "train",
            }
        }
    }
}

```

Train stages
* processor->dataset->dataloader->trainer
  1. processor: 


## Config Example

```hjson
    {
        "process": [
            {
                "_name": "space_tokenizer"
                "stages": ["train", "predict", "online"],
                "config": {
                    "data_pair": {
                        "origin": "origin_tokens"
                    }, // string or list, to_data[input['data'][data_set[..]]][to_data]=fn(input['data'][data_set[..]][from_data])
                    "data_set": ['train', 'dev'],
                    "map_char_token_idx": "origin_char_token_idx_map" // if this is empty string, will not do this
                }
            }, //0 the process num
            {
                "_name": "token_gather"
                "stages": ["train", "predict", "online"],
                "config": {
                    "tokens": "origin", // string or list, gather all filed
                    "deliver": "all_tokens",
                    "data_set": ['train', 'dev']
                }
            }, //1
            {
                "_name": "token_gather",
                "stages": ["train"],
                "config":{
                    "tokens": "label",
                    "deliver": "all_labels",
                    "data_set": ['train', 'dev'],
                    "deliver_method": "create_map"
                }
            }, //2
            {
                "_name": "label_to_id",
                "stages": ["train", "predict", "online"],
                "config": {
                    "data_pair": {
                        "label": "label_id"
                    }
                    "from_data": ["label"],
                    "to_data": ["label_id"],
                    "data_set": ['train', 'dev'],
                }
            }, //3
            {
                "_name": "get_static_embedding",
                "stages": ["train"],
                "config": {
                    "all_tokens": $ref,
                    "restructure_embedding_id": true
                    "deliver": "embedding",
                    "deliver_method": "new"
                }
            }, //4

            {
                "_name": "token_to_id",
                "stages": ["train", "predict", "online"],
                "config": {
                    "token_ids": $ref,
                    "token": "origin_tokens",
                    "ids": "origin_token_ids",
                    "data_set": ['train', 'dev'],
                }
            }, //5
        ],

        "save": {
            "train": { // stage, use "train.predict" means "train" & "predict"
                "data.train": "./train.pkl",
                "data.dev": "./dev.pkl",
                "token_ids": "./token_ids.pkl",
                "embedding": "./embedding.pkl",
                "label_ids": "./label_ids.pkl",
            },
            "predict": {
                "data.predict": "./predict.pkl"
            }
        },
        
        "load": {
            "predict.online":{
                "token_ids": "./token_ids.pkl",
                "embedding": "./embedding.pkl",
                "label_ids": "./label_ids.pkl",
            }
        },
        "_link": {
            "process.0.config.save_tokens":"process.2.config.tokens",
            "process.2.config.token_ids":"process.-1.config.token_ids",
        }
    }
```

## To Process Data Format Example

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
    "all_tokens": [..],
    "embedding": [..],
    "token_id_map": {"token": "id"},
    "id_token_map": {"id": "token"},
    "id_label_map": {"id": "label"},
    "label_id_map": {"label": "id"}
}
```
