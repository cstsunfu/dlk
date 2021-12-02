from dlkit.utils.vocab import Vocabulary
from dlkit.utils.config import ConfigTool
from typing import Dict, Callable, Set, List
from dlkit.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from functools import partial
import pandas as pd
from dlkit.utils.logger import logger

logger = logger()

@subprocessor_config_register('ner_prepro')
class NerPreProConfig(object):
    """docstring for NerPreProConfig
        {
            "_name": "ner_prepro",
            "config": {
                "train":{ //train、predict、online stage config,  using '&' split all stages
                    "data_set": {                   // for different stage, this processor will process different part of data
                        "train": ['train', 'valid', 'test'],
                        "predict": ['predict'],
                        "online": ['online']
                    },
                    "output_map": {   // without necessery don't change this
                        "sentence": "sentence",
                        "uuid": "uuid",
                        "entities_info": "entities_info",
                    },
                }, //3
                "predict": "train",
                "online": "train",
            }
        }

        NOTE: the nerformat input is
        {
            "uuid": '**-**-**-**'
            "sentence": "I have an apple",
            "labels": [
                        {
                            "end": 15,
                            "start": 10,
                            "labels": [
                                "Fruit"
                            ]
                        },
                        ...,
                    ]
                },
            ],
        }
    """
    def __init__(self, stage, config: Dict):

        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])
        self.output_map = self.config.get('output_map', {})


@subprocessor_register('ner_prepro')
class NerPrePro(ISubProcessor):
    """docstring for NerPrePro
    """

    def __init__(self, stage: str, config: NerPreProConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.data_set = config.data_set


    def process(self, data: Dict)->Dict:
        '''
            data: {
                "train": list of json format train data
            }
        '''

        if not self.data_set:
            return data

        for data_set_name in self.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip do ner_prepro on it.')
                continue
            data_set = data['data'][data_set_name]

            sentences = []
            uuids = []
            entities_infos = []

            for one_ins in data_set:
                if not one_ins['sentence'] or not one_ins['uuid'] or not one_ins['entities_info']:
                    logger.info(f"skip: {one_ins}")
                    continue
                sentences.append(one_ins['sentence'])
                uuids.append(one_ins['uuid'])
                entities_infos.append(one_ins['entities_info'])
            data_df = pd.DataFrame(data= {
                self.config.output_map["sentence"]: sentences,
                self.config.output_map["uuid"]: uuids,
                self.config.output_map["entities_info"]: entities_infos,
            })
            data['data'][data_set_name] = data_df

        return data
