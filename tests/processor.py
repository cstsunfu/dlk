import sys
import pandas as pd
sys.path.append('../')

from dlk.processors import PROCESSOR_CONFIG_REGISTRY, PROCESSOR_REGISTRY
# from dlk.models import DECODER_CONFIG_REGISTRY


config= {
    'data_set': {
        'train': ['train', 'dev'],
        'predict': ['predict'],
        'online': ['online']
    },
    'config_path': '../dlk/processors/wp_token.json',
    "normalizer": ['nfd', 'lowercase', 'strip_accents'],
    "pre_tokenizer": [{"whitespace": {}}],
    'post_tokenizer': 'bert',
    "filed_map": {
        "tokens": "tokens",
        "ids": "ids",
        "attention_mask": "attention_mask",
        "type_ids": "type_ids",
        "special_tokens_mask": "special_tokens_mask",
        "offsets": "offsets",
    },
    "data_type": "single",
    "process_data": [
        ['sentence', { "is_pretokenizerd": False}],
    ],
    # "data_type": "pair",
    # "process_data": [
        # ['sentence_a', { "is_pretokenizerd": false}],
        # ['sentence_b', {}],
    # ],
}
    # def __init__(self, parallel, status, **kwargs):
process_config = PROCESSOR_CONFIG_REGISTRY.get('wordpiece_tokenizer')(True, "online", **config)
process = PROCESSOR_REGISTRY.get('wordpiece_tokenizer')("online", process_config)

data = {}
data['data'] = {}
data['data']['online']

# df =

    # def process(self, data: Dict)->Dict:
        # for data_set_name in self.data_set:
            # data_set = data['data'][data_set_name]
            # data_set = self._process(data_set, self.process_data, self.filed_map)
            # data['data'][data_set_name] = data_set
        # return data
