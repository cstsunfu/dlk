"""
    This script is for reconvert the json format data to bio data to use conlleval script to get the score
"""
import json
from tokenizers import Token, Tokenizer
tokenizer = Tokenizer.from_file('../../dlk/local_data/embeddings/glove_tokenizer.json')

data = json.load(open('./predict.json', 'r'))


def align(sentence, labels):
    """TODO: Docstring for aligh.
    :sentence: TODO
    :labels: TODO
    :returns: TODO
    """
    # predicts = line['predict_entities_info']
    encode = tokenizer.encode(sentence)
    offsets = encode.offsets
    tokens = encode.tokens
    num_label = len(labels)
    token_num = len(offsets)
    output = []
    cur_token = 0
    cur_label = 0
    while cur_token<token_num:
        if cur_label >= num_label:
            output.append((tokens[cur_token], 'O'))
            cur_token += 1
            continue
        label_start = labels[cur_label]['start']
        label_end = labels[cur_label]['end']
        label = labels[cur_label]['labels'][0]

        token_start = offsets[cur_token][0]
        token_end = offsets[cur_token][1]

        if token_start==label_start:
            output.append((tokens[cur_token], 'B-'+label))
            cur_token += 1
        elif token_start<label_start:
            output.append((tokens[cur_token], 'O'))
            cur_token += 1
        elif token_start<label_end:
            output.append((tokens[cur_token], 'I-'+label))
            cur_token += 1
        elif token_start>=label_end:
            cur_label += 1
        else:
            raise PermissionError
    # # for truth in truthes:
    return output

write = []
for line in data:
    sentence = line['sentence']
    truth_labels = line['entities_info']
    predict_labels = line['predict_entities_info']
    truth = align(sentence, truth_labels)
    predict = align(sentence, predict_labels)
    one = []
    for t, p in zip(truth, predict):
        assert t[0] == p[0]
        one.append([t[0], t[1], p[1]])
    write.append(one)

with open('predict.txt', 'w') as f:
    for line in write:
        for token in line:
            f.write(' '.join(token)+'\n')
        f.write('\n')
