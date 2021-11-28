import json

token = json.load(open('./xlmr_tokenizer.json', 'r'))
token = json.load(open('../../../data/tokenizer.json', 'r'))
    # "model": {
        # "unk_id": 3,
        # "vocab": [
            # [
                # "<s>",
                # 0.0
            # ],
            # [
                # "<pad>",
                # 0.0
            # ],
            # [
                # "</s>",
                # 0.0
            # ],
            # [
                # "<unk>",
                # 0.0
            # ],
            # [
                # ",",
                # -3.4635426998138428
            # ],
            # [
                # ".",
                # -3.625642776489258
            # ],
            # [
                # "▁",
                # -3.9299705028533936
            # ],
            # [
                # "s",
                # -5.072621822357178
            # ],
json.dump(token, open('xbert_indent_token.json', 'w'), ensure_ascii=False, indent=4)
    # "added_tokens": [
        # {
            # "id": 0,
            # "special": true,
            # "content": "<s>",
            # "single_word": false,
            # "lstrip": false,
            # "rstrip": false,
            # "normalized": false
        # },
        # {
            # "id": 1,
            # "special": true,
            # "content": "<pad>",
            # "single_word": false,
            # "lstrip": false,
            # "rstrip": false,
            # "normalized": false
        # },
        # {
            # "id": 2,
            # "special": true,
            # "content": "</s>",
            # "single_word": false,
            # "lstrip": false,
            # "rstrip": false,
            # "normalized": false
        # },
        # {
            # "id": 3,
            # "special": true,
            # "content": "<unk>",
            # "single_word": false,
            # "lstrip": false,
            # "rstrip": false,
            # "normalized": false
        # },
        # {
            # "id": 250001,
            # "special": true,
            # "content": "<mask>",
            # "single_word": false,
            # "lstrip": false,
            # "rstrip": false,
            # "normalized": false
        # }
    # ],
