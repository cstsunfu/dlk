"""
    This file convert the conll03 format bio sequence labeling data to json format
    result is format as
    [{
        "uuid": '**-**-**-**'
        "sentence": "Mie Merah - Buah Bit - Noodles only - 1 box isi 4 - AM MA Kitchen - Red healthy noodle",
        "entities_info": [
                    {
                        "end": 65,
                        "start": 52,
                        "labels": [
                            "Brand"
                        ]
                    },
                    {
                        "end": 36,
                        "start": 22,
                        "labels": [
                            "Product"
                        ]
                    },
                    {
                        "end": 9,
                        "start": 0,
                        "labels": [
                            "Product"
                        ]
                    },
                    {
                        "end": 49,
                        "start": 38,
                        "labels": [
                            "Specification"
                        ]
                    },
                    {
                        "end": 86,
                        "start": 68,
                        "labels": [
                            "Health Benefits"
                        ]
                    }
                ]
            },
        ],
    },
    ]
"""
from tqdm import tqdm
import uuid
import json
import os


class Conll03Reader:
    def read(self, data_path):
        data_parts = ['train', 'valid', 'test']
        extension = '.txt'
        dataset = {}
        for data_part in tqdm(data_parts):
            file_path = os.path.join(data_path, data_part+extension)
            dataset[data_part] = self.read_file(str(file_path))

        format_dataset = {}
        for key, data in dataset.items():
            format_dataset[key] = self.convert(data)
        return format_dataset

    def convert(self, data):
        """TODO: Docstring for convert.
        :data: TODO
        :returns: TODO
        """
        format_data = []
        for line in data:
            tokens, labels = line[0], line[1]
            text = ""
            cur_label = ''
            start = -1
            entities_info = []
            entity_info = {}
            for token, label in zip(tokens, labels):
                assert label[0] in ['B', 'I', "O"]
                if label[0] == 'B':
                    if text:
                        cur_start = len(text) + 1 # will add space begin current token
                        text = " ".join([text, token])
                    else:
                        cur_start = 0
                        text = token
                    if cur_label:
                        entity_info['start'] = start
                        entity_info['end'] = len(text)
                        entity_info["labels"] = [cur_label]
                        entities_info.append(entity_info)
                    start = cur_start
                    cur_label = label.split('-')[-1]
                    entity_info = {}
                elif label[0] == 'O':
                    if cur_label:
                        entity_info['start'] = start
                        entity_info['end'] = len(text)
                        entity_info["labels"] = [cur_label]
                        entities_info.append(entity_info)
                    entity_info = {}
                    cur_label = ''
                    start = -1
                    if text:
                        text = " ".join([text, token])
                    else:
                        text = token
                else:
                    if text:
                        text = " ".join([text, token])
                    else:
                        text = token
            if cur_label:
                entity_info['start'] = start
                entity_info['end'] = len(text)
                entity_info['labels'] = [cur_label]
                entities_info.append(entity_info)
            for entity in entities_info:
                assert len(text[entity['start']: entity['end']].strip()) == entity['end'] - entity['start'], f"{entity}, {len(text[entity['start']: entity['end']].strip())},{entity['end'] - entity['start']},{text}"

            format_data.append({'uuid': str(uuid.uuid1()),  "sentence": text, "entities_info": entities_info})
        return format_data

    def read_file(self, file_path):
        samples = []
        tokens = []
        tags = []
        with open(file_path,'r', encoding='utf-8') as fb:
            for line in fb:
                line = line.strip('\n')

                if line == '-DOCSTART- -X- -X- O':
                    pass
                elif line =='':
                    if len(tokens) != 0:
                        samples.append((tokens, tags))
                        tokens = []
                        tags = []
                else:
                    contents = line.split(' ')
                    tokens.append(contents[0])
                    tags.append(contents[-1])
        return samples

if __name__ == "__main__":
    ds_rd = Conll03Reader()
    base_dir = "./data"
    data = ds_rd.read(base_dir)
    for key in data:
        json.dump(data[key], open(os.path.join(base_dir, key+'.json'), 'w'), indent=4)
