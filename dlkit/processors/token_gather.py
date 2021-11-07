from dlkit.utils.vocab import Vocabulary
from dlkit.utils.config import Config
from typing import Dict, Callable, Set, List
from dlkit.processors import processor_register, processor_config_register, Processor


@processor_config_register('token_gather')
class TokenGatherConfig(Config):
    """Config eg.
        {
            '_name': 'token_gather'
            'train': { // only train stage using
                'data_set': {                   // for different stage, this processor will process different part of data
                    'train': ['train', 'dev']
                },
                'gather_columns': ['label'], //List of columns. Every cell must be sigle token or list of tokens or set of tokens
                "deliver": "label_vocab", // output Vocabulary object (the Vocabulary of labels) name. 
                "update": null, // null or another Vocabulary object to update
            },
        }, 
    """

    def __init__(self, stage, **kwargs):
        self.data_set = kwargs.pop('data_set', {}).pop(stage, [])
        self.gather_columns = kwargs.pop("gather_column", [])
        self.deliver = kwargs.pop("deliver", "")
        if not self.deliver:
            raise ValueError("The 'deliver' value must not be null.")
        self.update = kwargs.pop('update', "")

@processor_register('token_gather')
class TokenGather(Processor):
    """
    """

    def __init__(self, stage: str, config: TokenGatherConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.data_set = config.data_set

    def process(self, data: Dict)->Dict:
        if self.config.update:
            self.vocab = data[self.config.update]
        else:
            self.vocab = Vocabulary(do_strip=True)
        for data_set_name in self.data_set:
            data_set = data['data'][data_set_name]
            for column in self.config.gather_columns:
                for line in data_set[column]:
                    self.vocab.auto_update(line)
        data[self.config.deliver] = self.vocab
        return data
