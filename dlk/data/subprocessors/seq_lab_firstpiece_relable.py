"""
Relabel the json data to bio.
And gather the firstpiece subword index to deliver $gather_index; for get the firstpiece index we must use the $word_ids(generator by tokenizer)
"""
from dlk.utils.vocab import Vocabulary
from dlk.utils.config import BaseConfig, ConfigTool
from typing import Dict, Callable, Set, List
from dlk.data.subprocessors import subprocessor_register, subprocessor_config_register, ISubProcessor
from functools import partial
from dlk.utils.logger import Logger

logger = Logger.get_logger()


@subprocessor_config_register('seq_lab_firstpiece_relabel')
class SeqLabFirstPieceRelabelConfig(BaseConfig):
    """docstring for SeqLabFirstPieceRelabelConfig
        {
            "_name": "seq_lab_firstpiece_relabel",
            "config": {
                "train":{ //train、predict、online stage config,  using '&' split all stages
                    "input_map": {  // without necessery, don't change this
                        "word_ids": "word_ids",
                        "offsets": "offsets",
                        "entities_info": "entities_info",
                    },
                    "data_set": {                   // for different stage, this processor will process different part of data
                        "train": ['train', 'valid', 'test', 'predict'],
                        "predict": ['predict'],
                        "online": ['online']
                    },
                    "output_map": {
                        "labels": "labels",
                        "gather_index": "gather_index",
                        "word_word_ids": "word_ids",
                        "word_offsets": "offsets",
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

        super(SeqLabFirstPieceRelabelConfig, self).__init__(config)
        self.config = ConfigTool.get_config_by_stage(stage, config)
        self.data_set = self.config.get('data_set', {}).get(stage, [])
        if not self.data_set:
            return
        self.word_ids = self.config['input_map']['word_ids']
        self.word_word_ids = self.config['output_map']['word_word_ids']
        self.word_offsets = self.config['output_map']['word_offsets']
        self.offsets = self.config['input_map']['offsets']
        self.entities_info = self.config['input_map']['entities_info']
        self.start_label = self.config['start_label']
        self.end_label = self.config['end_label']
        self.gather_index = self.config['output_map']['gather_index']
        self.output_labels = self.config['output_map']['labels']
        self.post_check(self.config, used=[
            "input_map",
            "data_set",
            "output_map",
            "start_label",
            "end_label",
        ])


@subprocessor_register('seq_lab_firstpiece_relabel')
class SeqLabFirstPieceRelabel(ISubProcessor):
    """docstring for SeqLabFirstPieceRelabel
    """

    def __init__(self, stage: str, config: SeqLabFirstPieceRelabelConfig):
        super().__init__()
        self.stage = stage
        self.config = config
        self.data_set = config.data_set
        if not self.data_set:
            logger.info(f"Skip 'seq_lab_firstpiece_relabel' at stage {self.stage}")
            return

    def process(self, data: Dict)->Dict:

        if not self.data_set:
            return data

        for data_set_name in self.data_set:
            if data_set_name not in data['data']:
                logger.info(f'The {data_set_name} not in data. We will skip do seq_lab_firstpiece_relabel on it.')
                continue
            data_set = data['data'][data_set_name]
            data_set[[self.config.output_labels,
                self.config.gather_index,
                self.config.word_word_ids,
                self.config.word_offsets]] = data_set.parallel_apply(self.relabel, axis=1, result_type="expand")
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

        gather_index = []
        pre_word_id = -1
        word_offset = []
        word_offsets = []
        word_ids = []
        for i, (token_offset, word_id) in enumerate(zip(offsets, sub_word_ids)):
            if word_id != pre_word_id:
                gather_index.append(i)
                word_ids.append(word_id)
                if word_offset:
                    word_offsets.append(word_offset)
                word_offset = list(token_offset)
                pre_word_id = word_id
            else:
                assert word_offset
                word_offset[1] = token_offset[1]
        if word_offset:
            word_offsets.append(word_offset)

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

        cur_token_index = 0
        offset_length = len(word_offsets)
        sub_labels = []
        for entity_info in entities_info:
            start_token_index = self.find_in_tuple(entity_info['start'], word_offsets, word_ids, cur_token_index, offset_length, is_start=True)
            if start_token_index == -1:
                logger.warning(f"cannot find the entity_info : {entity_info}, word_offsets: {word_offsets} ")
                continue
            for _ in range(start_token_index-cur_token_index):
                sub_labels.append('O')
            end_token_index = self.find_in_tuple(entity_info['end']-1, word_offsets, word_ids, start_token_index, offset_length)
            assert end_token_index != -1, f"entity_info: {entity_info}, word_offsets: {word_offsets}"
            sub_labels.append("B-"+entity_info['labels'][0])
            for _ in range(end_token_index-start_token_index):
                sub_labels.append("I-"+entity_info['labels'][0])
            cur_token_index = end_token_index + 1
        assert cur_token_index<=offset_length
        for _ in range(offset_length-cur_token_index):
            sub_labels.append('O')

        if word_ids[0] is None:
            sub_labels[0] = self.config.start_label

        if word_ids[offset_length-1] is None:
            sub_labels[offset_length-1] = self.config.end_label

        if len(sub_labels)!= len(gather_index):
            logger.error(f"{len(sub_labels)} vs {len(gather_index)}")
            for i in one_ins:
                logger.error(f"{i}")
            raise PermissionError

        return sub_labels, gather_index, word_ids, word_offsets
