import hjson
from tokenizers import Tokenizer
from typing import Dict
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='./convert.hjson',
        help="The config path.",
    )

    parser.add_argument(
        "--base_tokenizer_config",
        type=str,
        default='./tokenizer.json',
        help="The base wordpiece tokenizer config path for from embedding/vocab file convert to tokenizer.json config for tokenizers.",
    )

    parser.add_argument(
        "--type", type=str, default='vocab', help="vocab, embedding", choices=['vocab', "embedding"]
    )

    parser.add_argument("--output", type=str, default='./tokenizer.json', help="transformers tokenizer config files")


    args = parser.parse_args()
    if args.config:
        config_json = hjson.load(open(args.config), object_pairs_hook=dict)
        for key, value in config_json.items():
           setattr(args, key, value) 

    tokenizer = Tokenizer.from_file(args.tokenizer_config)
    # print(tokenizer.padding)

    if args.truncation:
        tokenizer.enable_truncation(args.truncation, stride=0, strategy=args.truncation_strategy)
    assert len(args.tokenize_output_files) == len(args.tokenize_files)
    for inp, out in zip(args.tokenize_files, args.tokenize_output_files):
        with open(inp, 'r') as f:
            lines = f.readlines()
        tokens = tokenizer.encode_batch(lines, is_pretokenized=False)
        with open(out, 'w') as f:
            for token in tokens:
                f.write(' '.join(token.tokens)+'\n')
