from typing import Dict, List
import uuid

def convert(data: List[List])->List[Dict]:
    """convert from bio to json

    Args:
        data: [[[tokens...], [tags..]], ..]

    Returns: list of dict format
        [
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
        ]
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
