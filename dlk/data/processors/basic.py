import hjson
import pandas as pd
from typing import Union, Dict
from dlk.utils.parser import BaseConfigParser
from dlk.data.processors import IProcessor, processor_config_register, processor_register
from dlk.data.subprocessors import subprocessor_config_register, subprocessor_register
from dlk.utils.config import BaseConfig
from dlk.utils.logger import Logger

logger = Logger.get_logger()

@processor_config_register('basic')
class BasicProcessorConfig(BaseConfig):
    """docstring for BasicProcessorConfig
    config e.g.
    {
        // input should be {"train": train, "valid": valid, ...}, train/valid/test/predict/online etc, should be dataframe and must have a column named "sentence"
        "_name": "basic@test_text_cls",
        "config": {
            "feed_order": ["load", "tokenizer", "token_gather", "label_to_id", "token_embedding", "save"]
        },
        "subprocessor@load": {
            "_base": "load",
            "config":{
                "base_dir": "."
                "predict":{
                    "meta": "./meta.pkl",
                },
                "online": [
                    "predict", //base predict
                    {   // special config, update predict, is this case, the config is null, means use all config from "predict", when this is empty dict, you can only set the value to a str "predict", they will get the same result
                    }
                ]
            }
        },
        "subprocessor@save": {
            "_base": "save",
            "config":{
                "base_dir": "."
                "train":{
                    "processed": "processed_data.pkl", // all data
                    "meta": {
                        "meta.pkl": ['label_vocab'] //only for next time use
                    }
                },
                "predict": {
                    "processed": "processed_data.pkl",
                }
            }
        },
        "subprocessor@tokenizer":{
            "_base": "fast_tokenizer",
            "config": {
                "train": {
                    "config_path": "*@*",
                    "prefix": ""
                    "data_type": "single", // single or pair, if not provide, will calc by len(process_data)
                    "process_data": [
                        ["sentence", { "is_pretokenized": false}],
                    ],
                    "post_processor": "default"
                    "filed_map": { // this is the default value, you can provide other name
                        "ids": "input_ids",
                    }, // the tokenizer output(the key) map to the value
                },
                "predict": "train",
                "online": "train"
            }
        },
        "subprocessor@token_gather":{
            "_base": "token_gather",
            "config": {
                "train": { // only train stage using
                    "data_set": {      // for different stage, this processor will process different part of data
                        "train": ["train", "valid"]
                    },
                    "gather_columns": ["label"], //List of columns. Every cell must be sigle token or list of tokens or set of tokens
                    "deliver": "label_vocab", // output Vocabulary object (the Vocabulary of labels) name.
                }
            }
        },
        "subprocessor@label_to_id":{
            "_base": "token2id",
            "config": {
                "train":{ //train、predict、online stage config,  using '&' split all stages
                    "data_pair": {
                        "label": "label_id"
                    },
                    "data_set": {                   // for different stage, this processor will process different part of data
                        "train": ['train', 'valid', 'test'],
                        "predict": ['predict'],
                        "online": ['online']
                    },
                    "vocab": "label_vocab", // usually provided by the "token_gather" module
                }, //3
                "predict": "train",
                "online": "train",
            }
        },
        "subprocessor@token_embedding": {
            "_base": "token_embedding",
            "config":{
                "train": { // only train stage using
                    "embedding_file": "*@*",
                    "tokenizer": "*@*", //List of columns. Every cell must be sigle token or list of tokens or set of tokens
                    "deliver": "token_embedding", // output Vocabulary object (the Vocabulary of labels) name.
                    "embedding_size": 200,
                }
            }
        },
    }
    """

    def __init__(self, stage, config: Dict):
        super(BasicProcessorConfig, self).__init__(config)
        if isinstance(config, str):
            config = hjson.load(open(config), object_pairs_hook=dict)
        self.feed_order = config["config"]['feed_order']
        self.subprocessors = config
        self.stage = stage
        # self.post_check(config['config'], used=['feed_order'])


@processor_register('basic')
class BasicProcessor(IProcessor):
    """docstring for DataSet"""
    def __init__(self, stage: str, config: BasicProcessorConfig):
        super(BasicProcessor, self).__init__()
        self._name = "basic"
        self.stage = stage
        self.feed_order = config.feed_order
        assert len(self.feed_order) > 0

        self.subprocessors = {}
        for name in self.feed_order:
            subprocessor_config_dict = config.subprocessors[f'subprocessor@{name}']
            logger.info(f"Init '{name}' ....")
            subprocessor_name = subprocessor_config_dict["_name"]
            subprocessor_config = subprocessor_config_register.get(subprocessor_name)(stage=self.stage, config=subprocessor_config_dict)
            subprocessor = subprocessor_register.get(subprocessor_name)(stage=self.stage, config=subprocessor_config)
            self.subprocessors[name] = subprocessor

    def process(self, data: Dict)->Dict:
        """TODO: Docstring for process.

        :data: TODO
        :returns: TODO
        """
        logger.info(f"Start Data Processing....")
        for name in self.feed_order:
            if self.stage != 'online':
                logger.info(f"Processing on '{name}' ....")
            data = self.subprocessors[name].process(data)
        logger.info(f"Data Processed.")

        return data

