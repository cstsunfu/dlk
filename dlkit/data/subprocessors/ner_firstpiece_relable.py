from dlkit.utils.vocab import Vocabulary
from dlkit.utils.config import ConfigTool
from typing import Dict, Callable, Set, List
from dlkit.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from functools import partial
from dlkit.utils.logger import logger

logger = logger()

@subprocessor_config_register('ner_firstpiece_relabel')
class NerFirstpieceRelabelConfig(object):
    """docstring for NerFirstpieceRelabelConfig
        {
            "_name": "ner_firstpiece_relabel",
            "config": {
                "train":{ //train、predict、online stage config,  using '&' split all stages
                    "input_map": {  // without necessery, don't change this
                        "word_ids": "word_ids",
                        "offsets": "offsets",
                        "entities_info": "entities_info",
                    },
                    "data_set": {                   // for different stage, this processor will process different part of data
                        "train": ['train', 'valid', 'test'],
                        "predict": ['predict'],
                        "online": ['online']
                    },
                    "output_map": {
                        "labels": "labels",
                        "gather_index": "gather_index",
                    },
                    "start_label": "S",
                    "end_label": "E",
                }, //3
                "predict": "train",
                "online": "train",
            }
        }
    """
    def __init__(self, stage, config: Dict):

        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])
        self.word_ids = self.config['input_map']['word_ids']
        self.offsets = self.config['input_map']['offsets']
        self.entities_info = self.config['input_map']['entities_info']
        self.start_label = self.config['start_label']
        self.end_label = self.config['end_label']
        self.output_labels = self.config['output_map']['labels']
        self.gather_index = self.config['output_map']['gather_index']


@subprocessor_register('ner_firstpiece_relabel')
class NerFirstpieceRelabel(ISubProcessor):
    """docstring for NerFirstpieceRelabel
    """

    def __init__(self, stage: str, config: NerFirstpieceRelabelConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.data_set = config.data_set

    def process(self, data: Dict)->Dict:

        if not self.data_set:
            return data

        for data_set_name in self.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip do ner_firstpiece_relabel on it.')
                continue
            data_set = data['data'][data_set_name]
            # data_set[[self.config.output_labels, self.config.gather_index]] = data_set.parallel_apply(self.relabel, axis=1, result_type="expand")
            data_set[[self.config.output_labels, self.config.gather_index]] = data_set.apply(self.relabel, axis=1, result_type="expand")

        return data

    def find_in_tuple(self, key, tuple_list, sub_word_ids, start, length, is_start=False):
        """TODO: Docstring for find.

        :key: TODO
        :tuple_list: TODO
        :start: TODO
        :returns: TODO
        """
        while start<length:
            if sub_word_ids[start] is None:
                start += 1
            elif key>=tuple_list[start][0] and key<tuple_list[start][1]:
                return start
            elif key<tuple_list[start][0]:
                if is_start:
                    return -1
                else:
                    return start - 1
            else:
                start += 1
        return -1

    def relabel(self, one_ins):
        """TODO: Docstring for relabel.
        :returns: TODO
        """
        pre_clean_entities_info = one_ins[self.config.entities_info]
        pre_clean_entities_info.sort(key=lambda x: x['start'])
        offsets = one_ins[self.config.offsets]
        sub_word_ids = one_ins[self.config.word_ids]
        if not sub_word_ids:
            logger.warning(f"entity_info: {pre_clean_entities_info}, offsets: {offsets} ")

        entities_info = []
        pre_end = -1
        pre_length = 0
        for entity_info in pre_clean_entities_info:
            assert len(entity_info['labels']) == 1, f"currently we just support one label for one entity"
            if entity_info['start']<pre_end:
                if entity_info['end'] - entity_info['start'] > pre_length:
                    entities_info.pop()
                else:
                    continue
            entities_info.append(entity_info)
            pre_end = entity_info['end']
            pre_length = entity_info['end'] - entity_info['start']

        cur_word_index = 0
        offset_length = len(offsets)
        gather_index = []
        sub_labels = []
        cur_token_index = 0
        if sub_word_ids[0] is None:
            sub_labels.append('O')
            gather_index.append(0)
            cur_token_index = 1
        for entity_info in entities_info:
            start_token_index = self.find_in_tuple(entity_info['start'], offsets, sub_word_ids, cur_token_index, offset_length, is_start=True)
            if start_token_index == -1:
                logger.warning(f"cannot find the entity_info : {entity_info}, offsets: {offsets} ")
                continue
            start_word_index = sub_word_ids[start_token_index]
            for reserve_word_index in range(cur_word_index, start_word_index):
                sub_labels.append('O')
                while cur_token_index<offset_length:
                    if (cur_token_index==0 or cur_token_index==offset_length-1) and sub_word_ids[cur_token_index] is None:
                        gather_index.append(cur_token_index)
                        cur_token_index += 1
                        break
                    elif sub_word_ids[cur_token_index] < reserve_word_index:
                        cur_token_index += 1
                    elif sub_word_ids[cur_token_index] == reserve_word_index:
                        gather_index.append(cur_token_index)
                        cur_token_index += 1
                        break
                    else:
                        raise PermissionError(f"{reserve_word_index} is smaller than current index {sub_word_ids[cur_token_index]}, {sub_word_ids}")

            end_token_index = self.find_in_tuple(entity_info['end']-1, offsets, sub_word_ids, start_token_index, offset_length)
            assert end_token_index != -1, f"entity_info: {entity_info}, offsets: {offsets}"
            end_word_index = sub_word_ids[end_token_index]

            sub_labels.append("B-"+entity_info['labels'][0])
            gather_index.append(start_token_index)

            cur_word_index = start_word_index + 1
            cur_token_index = start_token_index + 1
            for reserve_word_index in range(cur_word_index+1, end_word_index):
                sub_labels.append("I-"+entity_info['labels'][0])
                while cur_token_index<offset_length:
                    if (cur_token_index==0 or cur_token_index==offset_length-1) and sub_word_ids[cur_token_index] is None:
                        gather_index.append(cur_token_index)
                        cur_token_index += 1
                        break
                    elif sub_word_ids[cur_token_index] < reserve_word_index:
                        cur_token_index += 1
                    elif sub_word_ids[cur_token_index] == reserve_word_index:
                        gather_index.append(cur_token_index)
                        cur_token_index += 1
                        break
                    else:
                        raise PermissionError(f"{reserve_word_index} is smaller than current index {sub_word_ids[cur_token_index]}, {sub_word_ids}")

            cur_token_index = end_token_index + 1
            cur_word_index = end_word_index + 1
        assert cur_token_index<=offset_length
        pre_word_index = sub_word_ids[cur_token_index-1]
        if pre_word_index is None:
            pre_word_index = -1
        for token_index in range(cur_token_index, offset_length):
            if sub_word_ids[token_index] == pre_word_index:
                continue
            elif (sub_word_ids[token_index] is None) or ((sub_word_ids[token_index] == pre_word_index+1)):
                gather_index.append(token_index)
                sub_labels.append("O")
                pre_word_index += 1
            else:
                raise PermissionError(f"The token_index is {token_index}, sub_word_ids is {sub_word_ids}, pre_word_index is {pre_word_index}")

        if sub_word_ids[0] is None:
            assert sub_labels[0] == 'O', sub_labels
            sub_labels[0] = self.config.start_label

        if sub_word_ids[offset_length-1] is None:
            assert sub_labels[-1] == 'O', sub_labels
            sub_labels[-1] = self.config.end_label

        word_num = 0
        word_index_set = set()
        for word_id in sub_word_ids:
            if word_id is None:
                word_num += 1
            elif word_id not in word_index_set:
                word_index_set.add(word_id)
                word_num += 1

        if len(sub_labels)!= word_num:
            logger.error(f"{len(sub_labels)} vs {word_num}")
            for i in one_ins:
                logger.error(f"{i}")
            raise PermissionError

        return sub_labels, gather_index

# 12/08/2021 18:57:35 - ERROR - dlkit - 3 vs 4
# 12/08/2021 18:57:35 - ERROR - dlkit - Peter Blackburn
# 12/08/2021 18:57:35 - ERROR - dlkit - 9531d934-53e9-11ec-ab17-acde48001122
# 12/08/2021 18:57:35 - ERROR - dlkit - [{'start': 0, 'end': 15, 'labels': ['PER']}]
# 12/08/2021 18:57:35 - ERROR - dlkit - ['<s>', 'Peter', 'Black', 'burn', '</s>']
# 12/08/2021 18:57:35 - ERROR - dlkit - [0, 22611, 11368, 7554, 2]
# 12/08/2021 18:57:35 - ERROR - dlkit - [1, 1, 1, 1, 1]
# 12/08/2021 18:57:35 - ERROR - dlkit - [0, 0, 0, 0, 0]
# 12/08/2021 18:57:35 - ERROR - dlkit - [1, 0, 0, 0, 1]
# 12/08/2021 18:57:35 - ERROR - dlkit - [(0, 0), (0, 5), (6, 11), (11, 15), (0, 0)]
# 12/08/2021 18:57:35 - ERROR - dlkit - [None, 0, 1, 1, None]
# 12/08/2021 18:57:35 - ERROR - dlkit - []
# 12/08/2021 18:57:35 - ERROR - dlkit - [None, 0, 0, 0, None]
