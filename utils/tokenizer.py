from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.models import BPE
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

VOCAB_SIZE = 2

trainer = WordPieceTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer = Tokenizer(WordPiece(vocab={"a": 0}, unk_token="[UNK]"))

tokenizer.pre_tokenizer = Whitespace()

files = [f"wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
files = [f"wikitext-103-raw/wiki.{split}.raw" for split in ["test"]]

tokenizer.train([], trainer)
tokenizer.save('tokenizer.json', pretty=True)

