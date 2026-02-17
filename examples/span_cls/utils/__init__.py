import uuid
from typing import Dict, List


def convert(data: List[List]) -> List[Dict]:
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
        entities_info = []

        current_entity = None  # {start, label}

        for token, label in zip(tokens, labels):
            start_idx = len(text) + 1 if text else 0
            end_idx = start_idx + len(token)

            if text:
                text += " " + token
            else:
                text = token

            tag_type = label[0]
            tag_value = label.split("-")[-1] if "-" in label else ""

            if tag_type == "B":
                if current_entity:
                    current_entity["end"] = start_idx - 1  # 减去当前的空格
                    entities_info.append(current_entity)
                current_entity = {"start": start_idx, "labels": [tag_value]}

            elif tag_type == "O":
                if current_entity:
                    current_entity["end"] = start_idx - 1
                    entities_info.append(current_entity)
                    current_entity = None

            elif tag_type == "I":
                if current_entity:
                    pass
                else:
                    current_entity = {"start": start_idx, "labels": [tag_value]}

        if current_entity:
            current_entity["end"] = len(text)
            entities_info.append(current_entity)

        for entity in entities_info:
            assert (
                len(text[entity["start"] : entity["end"]].strip())
                == entity["end"] - entity["start"]
            ), f"{entity}, {len(text[entity['start']: entity['end']].strip())},{entity['end'] - entity['start']},{text}"

        format_data.append(
            {
                "uuid": str(uuid.uuid4()),
                "sentence": text,
                "entities_info": entities_info,
            }
        )
    return format_data
