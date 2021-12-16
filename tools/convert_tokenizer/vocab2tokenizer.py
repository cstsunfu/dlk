import json
import hjson
import argparse

pre_tokenizers = {
    "bert": {"type": "BertPreTokenizer"},
    "whitespacesplit": {"type": "WhitespaceSplit"}
}

template = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {
            "id": 0,
            "special": True,
            "content": "[CLS]",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False
        },
    ],
    "normalizer": {
        "type": "BertNormalizer",
        "clean_text": True,
        "handle_chinese_chars": True,
        "strip_accents": None,
        "lowercase": True
    },
    "pre_tokenizer": {
        "type": "BertPreTokenizer"
    },
    "post_processor": {
        "type": "TemplateProcessing",
        "single": [
            {
                "SpecialToken": {
                    "id": "[CLS]",
                    "type_id": 0
                }
            },
            {
                "Sequence": {
                    "id": "A",
                    "type_id": 0
                }
            },
            {
                "SpecialToken": {
                    "id": "[SEP]",
                    "type_id": 0
                }
            }
        ],
        "pair": [
            {
                "SpecialToken": {
                    "id": "[CLS]",
                    "type_id": 0
                }
            },
            {
                "Sequence": {
                    "id": "A",
                    "type_id": 0
                }
            },
            {
                "SpecialToken": {
                    "id": "[SEP]",
                    "type_id": 0
                }
            },
            {
                "Sequence": {
                    "id": "B",
                    "type_id": 1
                }
            },
            {
                "SpecialToken": {
                    "id": "[SEP]",
                    "type_id": 1
                }
            }
        ],
        "special_tokens": {
            "[CLS]": {
                "id": "[CLS]",
                "ids": [
                    0
                ],
                "tokens": [
                    "[CLS]"
                ]
            },
            "[SEP]": {
                "id": "[SEP]",
                "ids": [
                    1
                ],
                "tokens": [
                    "[SEP]"
                ]
            }
        }
    },
    "decoder": {
        "type": "WordPiece",
        "prefix": "##",
        "cleanup": True
    },
    "model": {
        "type": "WordPiece",
        "unk_token": "[UNK]",
        "continuing_subword_prefix": "##",
        "max_input_chars_per_word": 100,
        "vocab": {
        }
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='./vocab2tokenizer.hjson',
        help="The config path.",
    )

    parser.add_argument(
        "--unk",
        type=str,
        default='[UNK]',
        help="The unk token repretation.",
    )

    parser.add_argument(
        "--vocab",
        type=str,
        default='vocab.txt',
        help="The vocab path.",
    )

    parser.add_argument(
        "--do_norm",
        type=bool,
        default=True,
        help="Do BertNormalizer or not.",
    )

    parser.add_argument(
        "--do_lowercase",
        type=bool,
        default=True,
        help="Normalize do lowercase or not.",
    )
    parser.add_argument(
        "--pre_tokenizer",
        type=str,
        default='bert',
        help="bert or whitespacesplit",
    )

    parser.add_argument(
        "--do_bert_postprocess",
        type=bool,
        default=True,
        help="Whether add the [CLS] and [SEP] token for single or pair input.",
    )

    parser.add_argument("--output", type=str, default='./vocab_tokenizer.json', help="generated transformers tokenizer config files")


    args = parser.parse_args()
    if args.config:
        config_json = hjson.load(open(args.config), object_pairs_hook=dict)
        for key, value in config_json.items():
           setattr(args, key, value)

    tokenizer = template

    added_tokens = ['[CLS]', '[SEP]', '[MASK]', "[PAD]"] + [args.unk]

    added_tokens_config = []
    vocab = {}
    for i, token in enumerate(added_tokens):
        added_templete = {
              "id": 0,
              "special": True,
              "content": "",
              "single_word": True,
              "lstrip": False,
              "rstrip": False,
              "normalized": False
        }
        added_templete['id'] = i
        added_templete['content'] = token
        added_tokens_config.append(added_templete)
        if token not in vocab:
            vocab[token] = i

    normalizer = {
        "type": "BertNormalizer",
        "clean_text": True,
        "handle_chinese_chars": True,
        "strip_accents": None,
        "lowercase": True if args.do_lowercase else False
    }

    with open(args.vocab, 'r') as f:
        lines = f.readlines()
        for line in lines:
            token = line.strip()
            if token not in vocab:
                vocab[token] = len(vocab)
    tokenizer['added_tokens'] = added_tokens_config
    tokenizer['model']['vocab'] = vocab
    tokenizer['model']['unk_token'] = args.unk
    tokenizer['normalizer'] = normalizer if args.do_norm else None
    tokenizer['pre_tokenizer'] = pre_tokenizers[args.pre_tokenizer]
    if not args.do_bert_postprocess:
        tokenizer['post_processor'] = None

    json.dump(tokenizer, open(args.output, 'w'), indent=4)
