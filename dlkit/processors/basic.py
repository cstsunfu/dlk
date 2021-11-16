import hjson
import pandas as pd
from typing import Union, Dict
from dlkit.utils.parser import BaseConfigParser
from dlkit.processors import IProcessor, processor_config_register, processor_register
from dlkit.subprocessors import subprocessor_config_register, subprocessor_register
from dlkit.utils.config import ConfigTool

@processor_config_register('basic')
class BasicProcessorConfig(object):
    """docstring for BasicProcessorConfig
    config e.g.
    {
        "processor": {
            "_name": "basic",
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
    """

    def __init__(self, stage, config: Dict):
        if isinstance(config, str):
            config = hjson.load(open(config), object_pairs_hook=dict)
        self.feed_order = config.pop("config", {}).pop('feed_order', [])
        self.subprocessors = config
        self.stage = stage


@processor_register('basic')
class BasicProcessor(IProcessor):
    """docstring for DataSet"""
    def __init__(self, stage: str, config: BasicProcessorConfig):
        super(BasicProcessor, self).__init__()
        self._name = "basic"
        self.stage = stage
        self.feed_order = config.feed_order
        assert len(self.feed_order) > 0
        self.subprocessors = config.subprocessors

    def process(self, data: Dict)->Dict:
        """TODO: Docstring for process.

        :data: TODO
        :returns: TODO
        """
        for subprocessor_name in self.feed_order:
            subprocessor_config_dict = self.subprocessors.get(f'subprocessor@{subprocessor_name}')
            subprocessor_name = subprocessor_config_dict.get("_name")
            subprocessor_config = subprocessor_config_register.get(subprocessor_name)(stage=self.stage, config=subprocessor_config_dict)
            subprocessor = subprocessor_register.get(subprocessor_name)(stage=self.stage, config=subprocessor_config)
            data = subprocessor.process(data)

        return data
