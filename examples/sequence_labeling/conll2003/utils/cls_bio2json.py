# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from . import convert
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
            format_dataset[key] = convert(data)
        return format_dataset


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
