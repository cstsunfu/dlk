import sys
import pandas as pd
# sys.path.append('../../../')

from dlk.processors import processor_register, processor_config_register
# from dlk.models import DECODER_CONFIG_REGISTRY


def test_wordpiece():
    """TODO: Docstring for test_test.
    :returns: TODO

    """
    config= {
        'data_set': {
            'train': ['train', 'dev'],
            'predict': ['predict'],
            'online': ['online']
        },
        'config_path': './tests/processors/wp_token.json',
        "normalizer": ['nfd', 'lowercase', 'strip_accents'],
        "pre_tokenizer": [{"whitespace": {}}],
        'post_processor': 'bert',
        "filed_map": {
            "tokens": "tokens",
            "ids": "ids",
            "attention_mask": "attention_mask",
            "type_ids": "type_ids",
            "special_tokens_mask": "special_tokens_mask",
            "offsets": "offsets",
        },
        # "data_type": "single",
        # "process_data": [
            # ['sentence', { "is_pretokenized": False}],
        # ],
        "data_type": "pair",
        "process_data": [
            ['sentence_a', { "is_pretokenized": False}],
            ['sentence_b', {}],
        ],
    }
    process_config = processor_config_register.get('wordpiece_tokenizer')("online", **config)
    process = processor_register.get('wordpiece_tokenizer')("online", process_config)

    data = {}
    data['data'] = {}
    data['data']['online'] = pd.DataFrame(data={"sentence_a":['Ni a', 'wo hen hao'], "sentence_b": ['1 2', '3 4']})
    process_column_data = data['data']['online'][['sentence_a', "sentence_b"]]
    # print(process_column_data)
    # print([pair for pair in process_column_data.values])
    # print(list(process_column_data.values))
    # for i in process_column_data.values:
        # print(i)
    data = process.process(data)
    for c in data['data']['online']:
        print(c)
        print(data['data']['online'][c])
# # print(data)
