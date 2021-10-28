from dlkit.utils.vocab import Vocabulary
from dlkit.utils.config import Config
from typing import Dict, Callable, Set, List
from dlkit.processors import processor_register, processor_config_register, Processor


@processor_config_register('token2id')
class Token2IDConfig(Config):
    """docstring for Token2IDConfig
        {
            "_name": "token2id",
            "_status": ["train", "predict", "online"],
            "config": {
                "data_pair": {
                    "label": "label_id"
                },
                "data_set": {                   // for different status, this processor will process different part of data
                    "train": ['train', 'dev'],
                    "predict": ['predict'],
                    "online": ['online']
                },
            }
        }, //3
    """

    def __init__(self, status, **kwargs):
        self.data_set = kwargs.pop('data_set', {}).pop(status, [])
        self.gather_columns = kwargs.pop("gather_column", [])
        self.deliver = kwargs.pop("deliver", "")
        if not self.deliver:
            raise ValueError("The 'deliver' value must not be null.")
        self.update = kwargs.pop('update', "")

@processor_register('token2id')
class Token2ID(Processor):
    """docstring for Token2ID
    """

    def __init__(self, status: str, config: Token2IDConfig):
        super().__init__(status, config)
        self.status = status
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
