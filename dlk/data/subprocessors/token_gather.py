"""
gather all tokens from the 'gather_columns' and deliver a vocab named 'token_vocab'
"""
from dlk.utils.vocab import Vocabulary
from dlk.utils.config import BaseConfig, ConfigTool
from typing import Dict, Callable, Set, List
from dlk.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from dlk.utils.logger import Logger

logger = Logger.get_logger()

@subprocessor_config_register('token_gather')
class TokenGatherConfig(BaseConfig):
    """Config eg.
        {
            "_name": "token_gather",
            "config": {
                "train": { // only train stage using
                    "data_set": {                   // for different stage, this processor will process different part of data
                        "train": ["train", "valid", 'test']
                    },
                    "gather_columns": "*@*", //List of columns. Every cell must be sigle token or list of tokens or set of tokens
                    "deliver": "*@*", // output Vocabulary object (the Vocabulary of labels) name.
                    "ignore": "", // ignore the token, the id of this token will be -1
                    "update": null, // null or another Vocabulary object to update
                    "unk": "[UNK]",
                    "pad": "[PAD]",
                    "min_freq": 1,
                    "most_common": -1, //-1 for all
                }
            }
        }
    """

    def __init__(self, stage: str, config: Dict):
        super(TokenGatherConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])
        if not self.data_set:
            return
        self.ignore = self.config['ignore']
        self.gather_columns = self.config["gather_columns"]
        self.deliver = self.config["deliver"]
        if self.data_set and (not self.deliver):
            raise ValueError("The 'deliver' value must not be null.")
        self.update = self.config['update']
        self.unk = self.config['unk']
        self.pad = self.config['pad']
        self.min_freq = self.config['min_freq']
        self.most_common = self.config['most_common']
        self.post_check(self.config, used=[
            "data_set",
            "gather_columns",
            "deliver",
            "ignore",
            "update",
            "unk",
            "pad",
            "min_freq",
            "most_common",
        ])

@subprocessor_register('token_gather')
class TokenGather(ISubProcessor):
    """
    """
    def __init__(self, stage: str, config: TokenGatherConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.data_set = config.data_set
        if not self.data_set:
            logger.info(f"Skip 'token_gather' at stage {self.stage}")
            return
        self.update = config.update

    def process(self, data: Dict)->Dict:
        if not self.data_set:
            return data
        if self.update:
            self.vocab = data[self.update]
        else:
            self.vocab = Vocabulary(do_strip=True, unknown=self.config.unk, ignore=self.config.ignore, pad=self.config.pad)
        for data_set_name in self.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip gather tokens from it.')
                continue
            data_set = data['data'][data_set_name]
            for column in self.config.gather_columns:
                self.vocab.auto_update(data_set[column])
        self.vocab.filter_rare(self.config.min_freq, self.config.most_common)
        logger.info(f"The Vocab Num is {self.vocab.word_num}")
        data[self.config.deliver] = self.vocab.__dict__
        return data
