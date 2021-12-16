import hjson
from tokenizers import Tokenizer
from typing import Dict
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='./tokenizer.hjson',
        help="The config path.",
    )

    parser.add_argument(
        "--tokenizer_config",
        type=str,
        default='./tokenizer.json',
        help="The tokenizer config path.",
    )

    parser.add_argument(
        "--truncation", type=int, default=0, help="truncation length"
    )

    parser.add_argument(
        "--truncation_strategy", type=str, default='longest_first', help="longest_first, only_first, only_second", choices=['longest_first', "only_first", 'only_second']
    )

    parser.add_argument(
        "--tokenize_files", type=list, default=[], help="tokenize files."
    )
    parser.add_argument(
        "--tokenize_output_files", type=list, default=[], help="tokenize output files."
    )

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
