from tokenizers import Tokenizer
import json
tokenizer = Tokenizer.from_file('./new_tokenizer.json')

print(tokenizer.encode('I love you').tokens)

str = tokenizer.to_str()
json.dump(json.loads(str), open('dump.json', 'w'), indent=4)
