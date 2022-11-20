# Copyright cstsunfu. All rights reserved.
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
data format
{
    "uuid": "",
    "sentence": "",
    "entities_info": [
        {
            "entity_id": "bd798da3-52a8-11ed-801c-18c04d299e80",
            "start":1,
            "end": 3,
            "labels": [
                "Brand"
            ]
        },
        {
            "entity_id": "bd7a2928-52a8-11ed-8b5c-18c04d299e80",
            "start":5,
            "end": 9,
            "labels": [
                "Product"
            ]
        }
    ],
    "relations_info": [
        {
            "labels": [
                "belong_to"
            ],
            "from": "bd7a2928-52a8-11ed-8b5c-18c04d299e80", # id of entity
            "to": "bd798da3-52a8-11ed-801c-18c04d299e80",
        }
    ]
}
"""
from datasets import load_dataset
from dlk.utils.logger import Logger
import copy
import json
import hjson
from dlk.process import Processor
from utils import convert
import uuid
logger = Logger('log.txt').get_logger()



def simple_tokenize(sentence, sep=' '):
    pass
    token_infos = []
    one = None
    for i, c in enumerate(sentence):
        if c!=  sep and not one:
            one = {
                "start": i,
                "end": i,
                "token": ""
            }
        if c == sep: 
            if (not one) or (not one['token']):
                continue
            token_infos.append(one)
            one = None
        else:
            one['token'] += c
            one['end'] += 1
    if one:
        token_infos.append(one)
    return token_infos

def get_all_entities_position(tokens, tags, sentence):
    entities_list = []
    entity_info = None
    assert len(tokens) == len(tags), f"{len(tokens)}!= {len(tags)}"
    for token, tag in zip(tokens, tags):
        if tag == 0:
            if not entity_info:
                continue
            entities_list.append(entity_info)
            entity_info = None
        if tag > 3:
            if entity_info:
                entities_list.append(entity_info)
            entity_info = token
        if tag in (1, 2, 3):
            entity_info['end'] = token['end']
            entity_info['token'] = sentence[entity_info['start']: entity_info['end']]
    if entity_info:
        entities_list.append(entity_info)
    return entities_list


def get_data():
    data = load_dataset(path="xiaobendanyn/nyt10")
    data = data['test'].to_dict()

    data_list = []
    no_regular_cnt = 0
    for entities, relations, sentence in zip(data['entities'], data['relations'], data['sentext']):
        tokens = simple_tokenize(sentence)
        entities_info_dict = {}
        relations_info = []
        flag = True
        for relation in relations:
            _, entity_type_1, entity_type_2, relation_type = relation['rtext'].split("/")
            entity_type_map = {
                relation['em1']: entity_type_1,
                relation['em2']: entity_type_2,
            }
            tags = relation['tags']
            assert len(tags) == len(tokens)
            position_entities = get_all_entities_position(tokens, tags, sentence)
            if len(position_entities) != 2:
                logger.warning(f"No regular:\n {json.dumps(position_entities, indent=4, ensure_ascii=False)}")
                flag = False
                continue
            entities_index = {}
            first = -1
            second = -1
            for position_entity in position_entities:
                assert position_entity['token'] in entity_type_map, f"{position_entity['token']} not in {entity_type_map.keys()},\n Relation: \n{json.dumps(relation, indent=4, ensure_ascii=False)}\n Entities: \n{json.dumps(entities, indent=4, ensure_ascii=False)}\nRelations: \n{json.dumps(relations, indent=4, ensure_ascii=False)}\nSentence: \n{json.dumps(relations, indent=4, ensure_ascii=False)} \nPosition Entities: \n{json.dumps(position_entities, indent=4, ensure_ascii=False)} "
                position_entity['labels'] = [entity_type_map[position_entity['token']]]
                position_entity_str = json.dumps(position_entity)
                if position_entity_str not in entities_info_dict:
                    entities_info_dict[position_entity_str] = str(uuid.uuid1())
                if position_entity['token'] == relation['em1']:
                    first = entities_info_dict[position_entity_str]
                else:
                    assert position_entity['token'] == relation['em2']
                    second = entities_info_dict[position_entity_str]
            relation_info = {
                    "labels": [relation_type],
                    "from": first,
                    "to": second
            }
            relations_info.append(relation_info)
        if not flag:
            no_regular_cnt += 1
            continue
        entities_info = []
        for entity_info, id in entities_info_dict.items():
            entity_info = json.loads(entity_info)
            entity_info.pop('token')
            entity_info['entity_id'] = id
            entities_info.append(entity_info)

        data_list.append({
            "uuid": str(uuid.uuid1()),
            "sentence": sentence,
            "entities_info": entities_info,
            "relations_info": relations_info
        })
    return data_list
# with open('./test.json', 'w') as f:
#     json.dump(get_data(), f, indent=4, ensure_ascii=False)
with open('./test.json', 'r') as f:
    data = json.load(f)

train = data[:len(data)//2]
valid = data[len(data)//2:]
input = {"data": {"train": valid, "valid": valid}}

processor = Processor('./bert/prepro.hjson')
processor.fit(input)
