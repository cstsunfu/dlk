# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import uuid

from datasets import load_dataset
from utils import convert

from dlk.preprocess import PreProcessor

label_map = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}


def convert_to_span_format(batch):
    output = {"sentence": [], "entities_info": [], "uuid": []}
    for tokens, tags in zip(batch["tokens"], batch["ner_tags"]):
        labels = [label_map[i] for i in tags]

        text = ""
        entities_info = []
        current_entity = None

        for token, label in zip(tokens, labels):
            start_idx = len(text) + 1 if text else 0
            if text:
                text += " " + token
            else:
                text = token

            tag_type = label[0]
            tag_value = label.split("-")[-1] if "-" in label else ""

            if tag_type == "B":
                if current_entity:
                    current_entity["end"] = start_idx - 1
                    entities_info.append(current_entity)
                current_entity = {"start": start_idx, "labels": [tag_value]}

            elif tag_type == "O":
                if current_entity:
                    current_entity["end"] = start_idx - 1
                    entities_info.append(current_entity)
                    current_entity = None

            elif tag_type == "I":
                if not current_entity:
                    current_entity = {"start": start_idx, "labels": [tag_value]}

        if current_entity:
            current_entity["end"] = len(text)
            entities_info.append(current_entity)

        output["sentence"].append(text)
        output["entities_info"].append(entities_info)
        output["uuid"].append(str(uuid.uuid4()))

    return output


if __name__ == "__main__":
    # Load Data
    data = load_dataset("conll2003")

    # Process using efficient dataset map
    columns_to_remove = data["train"].column_names
    processed_data = data.map(
        convert_to_span_format, batched=True, remove_columns=columns_to_remove
    )

    # Create Datasets
    input_data = {
        "train": processed_data["train"].select(range(100)),  # Demo size
        "valid": processed_data["validation"].select(range(100)),
    }

    processor = PreProcessor("./config/bert_firstpiece_lstm_crf/processor.jsonc")
    processor.fit(input_data)
