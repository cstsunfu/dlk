from datasets import load_dataset
from dlk.utils.logger import Logger
import copy
import json
import hjson
from dlk.process import Processor
import uuid

logger = Logger('log.txt')

def flat(data):
    """TODO: Docstring for flat.

    :data: TODO
    :returns: TODO
    """
    sentence_as = data['sentence_a']
    sentence_bs = data['sentence_b']
    uuids = data['uuid']
    labelses = data['labels']
    print(labelses)
    return [{'sentence_a': sentence_a, 'sentence_b': sentence_b, 'labels': [labels], "uuid": uuid} for sentence_a, sentence_b, labels, uuid in zip(sentence_as, sentence_bs, labelses, uuids)]

label_map = {0: "entails", 1: 'nor', 2: 'contradicts', -1: "remove"}

data = load_dataset("snli")
data = data.map(lambda one: {"sentence_a": one["hypothesis"], "sentence_b": one["premise"], 'labels': label_map[one['label']], "uuid": str(uuid.uuid1())}, remove_columns=['hypothesis', 'label', 'premise'])
data = data.filter(lambda one: one['labels'] in {'entails', 'nor', 'contradicts'})

input = {"data": {"train": flat(data['train'].to_dict()), 'valid': flat(data['validation'].to_dict()), 'test': flat(data['test'].to_dict())}}

processor = Processor('./distil_bert/prepro.hjson')
processor.fit(input)
