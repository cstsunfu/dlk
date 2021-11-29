import json
data = json.load(open('./tokenizer/indent_token.json', 'r'))
data = json.load(open('./wikitext-2/mbert_tokenizer.json', 'r'))
print(data['added_tokens'])
added_tokens = data['added_tokens']

id2token = {}
token2id = {}

for token in added_tokens:
    token_id, token = token['id'], token['content']
    id2token[token_id] = token
    token2id[token] = token_id
    
vocab = []
current_id = 0
for token in vocab:
    if token not in token2id:
        while current_id in id2token:
            current_id += 1
        id2token[current_id] = token
        token2id[token] = current_id
        current_id += 1
# assert current_id == len(token2id)
            
        
        
print(data['model']['vocab'])
for i in data:
    print(i)
# with open('../utils/tokenizer/indent_token.json', 'r') as f:

# || [{'id': 0, 'special': True, 'content': '[PAD]', 'single_word': False, 'lstrip': False, 'rstrip': False, 'normalized': False}, {'id': 100, 'special': True, 'content': '[UNK]', 'single_word': False, 'lstrip': False, 'rstrip': False, 'normalized': False}, {'id': 101, 'special': True, 'content': '[CLS]', 'single_word': False, 'lstrip': False, 'rstrip': False, 'normalized': False}, {'id': 102, 'special': True, 'content': '[SEP]', 'single_word': False, 'lstrip': False, 'rstrip': False, 'normalized': False}, {'id': 103, 'special': True, 'content': '[MASK]', 'single_word': False, 'lstrip': False, 'rstrip': False, 'normalized': False}]

# || version
# || truncation
# || padding
# || added_tokens
# || normalizer
# || pre_tokenizer
# || post_processor
# || decoder
# || model
