import hjson
from typing import Dict, Union, Callable, List
from dlkit.models.seq_label import SeqLabelConfig


def load_hjson_file(file_name: str) -> Dict:
    """load hjson file by file_name

    :file_name: TODO
    :returns: TODO

    """
    json_file = hjson.load(open(file_name), object_pairs_hook=dict)
    return json_file


# task_config = load_hjson_file('../configures/tasks/simple_ner.hjson')
