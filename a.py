import json

token = json.load(open('./tokenizer.json'))

with open('./token.json', 'w', encoding='utf8') as f:
    json.dump(token, f, indent=4, ensure_ascii=False)
