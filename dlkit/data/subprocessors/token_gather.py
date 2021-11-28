from dlkit.utils.vocab import Vocabulary
from dlkit.utils.config import ConfigTool
from typing import Dict, Callable, Set, List
from dlkit.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from dlkit.utils.logger import logger

logger = logger()

@subprocessor_config_register('token_gather')
class TokenGatherConfig(object):
    """Config eg.
        {
            "_name": "token_gather",
            "config": {
                "train": { // only train stage using
                    "data_set": {                   // for different stage, this processor will process different part of data
                        "train": ["train", "valid"]
                    },
                    "gather_columns": "*@*", //List of columns. Every cell must be sigle token or list of tokens or set of tokens
                    "deliver": "*@*", // output Vocabulary object (the Vocabulary of labels) name. 
                    "ignore": "", // ignore the token, the id of this token will be -1
                    "update": null, // null or another Vocabulary object to update
                    "unk": "",
                }
            }
        }
    """

    def __init__(self, stage, config):
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])

        self.ignore = self.config['ignore']
        self.gather_columns = self.config["gather_columns"]
        self.deliver = self.config["deliver"]
        if self.data_set and (not self.deliver):
            raise ValueError("The 'deliver' value must not be null.")
        self.update = self.config['update']
        self.unk = self.config['unk']

@subprocessor_register('token_gather')
class TokenGather(ISubProcessor):
    """
    """
    def __init__(self, stage: str, config: TokenGatherConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.data_set = config.data_set
        self.update = config.update

    def process(self, data: Dict)->Dict:
        if not self.data_set:
            return data
        if self.update:
            self.vocab = data[self.update]
        else:
            self.vocab = Vocabulary(do_strip=True, unknown=self.config.unk, ignore=self.config.ignore)
        for data_set_name in self.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip gather tokens from it.')
                continue
            data_set = data['data'][data_set_name]
            for column in self.config.gather_columns:
                self.vocab.auto_update(data_set[column])
        data[self.config.deliver] = self.vocab.__dict__
        return data
