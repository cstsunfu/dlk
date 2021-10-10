# from tokenizers import Tokenizer
# from tokenizers.models import WordPiece
# from tokenizers.models import Unigram
# from tokenizers.models import BPE

# from tokenizers import normalizers
# from tokenizers import pre_tokenizers
# from tokenizers.normalizers import Lowercase, NFD, StripAccents

# from tokenizers.pre_tokenizers import Whitespace, ByteLevel
# from tokenizers.processors import TemplateProcessing
# from tokenizers.trainers import WordPieceTrainer
# from tokenizers.trainers import UnigramTrainer
# from tokenizers.trainers import BpeTrainer

# METHOD = Unigram
# Trainer = UnigramTrainer

# bert_tokenizer = Tokenizer(METHOD())
# bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
# bert_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace()])
# bert_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([ByteLevel()])
# bert_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])

# bert_tokenizer.post_processor = TemplateProcessing(
    # single="[CLS] $A [SEP]",
    # pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    # special_tokens=[
        # ("[CLS]", 1),
        # ("[SEP]", 2),
    # ],
# )

# trainer = Trainer(
    # vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
# )

# files = [f"./readme.md"]
# bert_tokenizer.train(files, trainer)

# bert_tokenizer.save("uni_token.json", pretty=True)




from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
import json

# config = json.load(open('./wp_token.json', 'r', encoding='utf-8'))
# print(config['pre_tokenizer'])
# tokenizer = Tokenizer.from_file('./wp_token.json')
tokenizer = Tokenizer.from_file('./uni_token.json')
# tokenizer.pre_tokenizer = ByteLevel()
# tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])
print(tokenizer.post_processor)
token = tokenizer.encode_batch(["a b c d", ['b c', 'd e']])

# tokenizer.decoder = decoders.ByteLevel()
# print(eval("token.tokens"))
# print(len(eval("token.ids")))
# print(getattr(token, "ids"))
# print(getattr(token, "tokens"))
print(token[0].tokens)
print(token[1].tokens)
