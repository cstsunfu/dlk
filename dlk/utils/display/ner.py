# Copyright the author(s) of DLK.
# Copyright spark-nlp-display.
#
# This original source code is copied from spark-nlp-display and licensed under the Apache license.

import json
import os
import random
from typing import List

import numpy as np

import dlk.utils.display.style_utils as style_config
from dlk.utils.display.annotation import SpanAnnotation


class NerVisualizer:
    def __init__(self, ignore_labels: List = []):
        here = os.path.abspath(os.path.dirname(__file__))
        with open(
            os.path.join(here, "label_colors/ner.json"), "r", encoding="utf-8"
        ) as f_:
            self.label_colors = json.load(f_)
        self.ignore_labels = set(ignore_labels)

    # public function to get color for a label
    def get_label_color(self, label):
        """Returns color of a particular label

        Input: entity label <string>
        Output: Color <string> or <None> if not found
        """

        if str(label).lower() in self.label_colors:
            return self.label_colors[label.lower()]
        else:
            return None

    # private function for colors for display
    def __get_label(self, label):
        """Internal function to generate random color codes for missing colors

        Input: dictionary of entity labels and corresponding colors
        Output: color code (Hex)
        """
        if str(label).lower() in self.label_colors:
            return self.label_colors[label.lower()]
        else:
            # update it to fetch from git new labels
            r = lambda: random.randint(0, 200)
            return "#%02X%02X%02X" % (r(), r(), r())

    def set_label_colors(self, color_dict):
        """Sets label colors.
        input: dictionary of entity labels and corresponding colors
        output: self object - to allow chaining
        note: Previous values of colors will be overwritten
        """

        for key, value in color_dict.items():
            self.label_colors[key.lower()] = value
        return self

    # main display function
    def __display_ner(self, result, label_col, document_col, original_text):
        if original_text is None:
            original_text = result[document_col][0].result

        label_color = {}
        html_output = ""
        pos = 0

        sorted_labs_idx = np.argsort([int(i.begin) for i in result[label_col]])
        sorted_labs = [result[label_col][i] for i in sorted_labs_idx]

        for entity in sorted_labs:
            entity_type = entity.metadata["entity"].lower()
            if (entity_type not in label_color) and (
                entity_type not in self.ignore_labels
            ):
                label_color[entity_type] = self.__get_label(entity_type)

            begin = int(entity.begin)
            end = int(entity.end)
            if pos < begin and pos < len(original_text):
                white_text = original_text[pos:begin]
                html_output += '<span class="spark-nlp-display-others" style="background-color: white">{}</span>'.format(
                    white_text
                )
            pos = end + 1

            if entity_type in label_color:
                html_output += '<span class="spark-nlp-display-entity-wrapper" style="background-color: {}"><span class="spark-nlp-display-entity-name">{} </span><span class="spark-nlp-display-entity-type">{}</span></span>'.format(
                    label_color[entity_type],
                    original_text[begin : end + 1],  # entity.result,
                    entity.metadata["entity"],
                )
            else:
                html_output += '<span class="spark-nlp-display-others" style="background-color: white">{}</span>'.format(
                    original_text[begin : end + 1]
                )

        if pos < len(original_text):
            html_output += '<span class="spark-nlp-display-others" style="background-color: white">{}</span>'.format(
                original_text[pos:]
            )

        html_output += """</div>"""

        html_output = html_output.replace("\n", "<br>")

        return html_output

    def display(self, result):
        """Displays NER visualization.
        Inputs:
        result -- A Dataframe or dictionary.
        label_col -- Name of the column/key containing NER annotations.
        document_col -- Name of the column/key containing text document.
        original_text -- Original text of type 'str'. If specified, it will take precedence over 'document_col' and will be used as the reference text for display.
        Output: Visualization
        """
        display_result = {"entities_info": []}
        for entity in result["predict_entities_info"]:
            display_result["entities_info"].append(
                SpanAnnotation._from_dict(
                    {
                        "begin": entity["start"],
                        "end": entity["end"],
                        "metadata": {"entity": entity["labels"][0]},
                    }
                )
            )

        html_content = self.__display_ner(
            display_result,
            "entities_info",
            None,
            result["sentence"],
        )

        html_content_save = style_config.STYLE_CONFIG_ENTITIES + " " + html_content

        return html_content_save


if __name__ == "__main__":
    vis = NerVisualizer()
    result = {
        "sentence": "CRICKET - 中文字CESTERSHIRE TAKE OVER AT TOP AFTER INNINGS VICTORY .",
        "uuid": "1b764970-be58-11ec-841b-18c04d299e80",
        "entities_info": [{"start": 10, "end": 24, "labels": ["ORG"]}],
        "predict_entities_info": [
            {"start": 10, "end": 24, "labels": ["PER"]},
            {"start": 38, "end": 41, "labels": ["MISC"]},
        ],
        "predict_extend_return": {},
    }
    dis = vis.display(result)

    with open("/home/sun/Downloads/dis.html", "w") as f_:
        f_.write(dis)
