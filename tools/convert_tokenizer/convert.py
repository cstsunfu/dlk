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

import hjson
from tokenizers import Tokenizer
from typing import Dict
import argparse
import json
from transformers import RobertaTokenizerFast, BertTokenizerFast, DistilBertTokenizerFast
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


TOKENIZER_MAP = {
    "bert": BertTokenizerFast,
    "roberta": RobertaTokenizerFast,
    "distil_bert": DistilBertTokenizerFast
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='./convert.hjson',
        help="The config path.",
    )

    parser.add_argument(
        "--type", type=str, default='roberta', help="roberta, bert", choices=['roberta', "bert"]
    )

    parser.add_argument("--config_dir", type=str, default='./roberta/', help="transformers tokenizer config files")
    parser.add_argument("--output", type=str, default='./roberta_tokenizer.json', help="transformers tokenizer config files")


    args = parser.parse_args()
    if args.config:
        config_json = hjson.load(open(args.config), object_pairs_hook=dict)
        for key, value in config_json.items():
           setattr(args, key, value)


    tokenizer = TOKENIZER_MAP[args.type].from_pretrained(args.config_dir)
    # print(len(tokenizer._tokenizer.get_vocab()))
    str = tokenizer._tokenizer.to_str()
    # json.dump(json.loads(str), open(args.output, 'w'), indent=4)

    tokenizer._tokenizer.save(pretty=True, path=args.output)
