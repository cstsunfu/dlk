import hjson
import os
from typing import Dict, Union, Callable, List, Any
# from models import MODEL_REGISTRY, MODEL_CONFIG_REGISTRY
from dlk.utils.parser import CONFIG_PARSER_REGISTRY
import json

#TODO: parser应该返回完整的参数

def load_hjson_file(file_name: str) -> Dict:
    """load hjson file by file_name

    :file_name: TODO
    :returns: TODO

    """
    json_file = hjson.load(open(file_name), object_pairs_hook=dict)
    return json_file

def test_base():
    """TODO: Docstring for test_base.
    :returns: TODO

    """
    config_file = load_hjson_file('./tests/parser/test_base_inp.hjson')
    out = [json.dumps(conf, sort_keys=True) for conf in load_hjson_file('./tests/parser/test_base_out.hjson')['out']]
    out.sort()
    focus = config_file.pop('_focus', {})
    parser = CONFIG_PARSER_REGISTRY['task'](config_file)
    configs = parser.parser()
    config_names = []

    for possible_config in configs:
        config_name = []
        if not focus:
            config_name.append(str(possible_config))
        else:
            for source, to in focus.items():
                config_point = possible_config
                trace = source.split('.')
                for t in trace:
                    config_point = config_point[t]
                config_name.append(to+str(config_point))
        config_names.append('_'.join(config_name))
    assert len(config_names) == len(set(config_names))
    for conf in configs:
        print(json.dumps(conf, indent=4))
    configs = [json.dumps(conf, sort_keys=True) for conf in configs]
    configs.sort()
    assert configs == out
    # if len(config_names) != len(set(config_names)):
        # print(len(config_names))
        # print(len(set(config_names)))
        # for config, name in zip(configs, config_names):
            # print(json.dumps(config, indent=4))
            # print(name)
        # raise NameError('The config_names is not unique.')
    # for config, name in zip(configs, config_names):
        # print(json.dumps(config, indent=4))
        # print(name)

test_base()
# Train('simple_ner')
# Train('./dlk/configures/tasks/simple_ner.hjson')
# Train('lstm_linear_ner')

