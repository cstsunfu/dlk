from dlkit.utils.vocab import Vocabulary
from dlkit.utils.config import Config
from typing import Dict, Callable, Set, List
from dlkit.processors import processor_register, processor_config_register, Processor


@processor_config_register('token2id')
class Token2IDConfig(Config):
    """docstring for Token2IDConfig
        {
            "_name": "token2id",
            "train&predict&online":{ //train、predict、online stage config,  using '&' split all stages
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
        }
    """

    def __init__(self, stage, **kwargs):
        self.data_set = kwargs.pop('data_set', {}).pop(stage, [])
        self.data_pair = kwargs.pop('data_pair', {})
        if self.data_set and (not self.data_pair):
            raise ValueError("The 'data_pair' must not be null.")
        self.vocab = kwargs.pop('vocab', "")
        if not self.vocab:
            raise ValueError("The 'vocab' must be provided.")


@processor_register('token2id')
class Token2ID(Processor):
    """docstring for Token2ID
    """

    def __init__(self, stage: str, config: Token2IDConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.data_set = config.data_set

    def process(self, data: Dict)->Dict:
        self.vocab = data[self.config.vocab]
        for data_set_name in self.data_set:
            data_set = data['data'][data_set_name]
            for column in self.config.gather_columns:
                for line in data_set[column]:
                    self.vocab.auto_update(line)
        data[self.config.deliver] = self.vocab
        return data
