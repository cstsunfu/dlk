import hjson
from tokenizers import Tokenizer
from typing import Dict
import argparse
from transformers import RobertaTokenizerFast, BertTokenizerFast

TOKENIZER_MAP = {
    "bert": BertTokenizerFast,
    "roberta": RobertaTokenizerFast
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
    tokenizer._tokenizer.save(pretty=True, path=args.output)
