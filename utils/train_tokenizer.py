import hjson
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
import argparse


hjson.load()
json_file = hjson.load(open(file_name), object_pairs_hook=dict)

VOCAB_SIZE = 2

trainer = WordPieceTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

tokenizer = Tokenizer(WordPiece(vocab={"a": 0}, unk_token="[UNK]"))

tokenizer.pre_tokenizer = Whitespace()

files = [f"wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
files = [f"wikitext-103-raw/wiki.{split}.raw" for split in ["test"]]

tokenizer.train([], trainer)
tokenizer.save('tokenizer.json', pretty=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="The config path.",
    )
    parser.add_argument(
        "--train_files", type=list, default=None, help="train tokenizer files."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000,
        help=( "vocab size")
    )
    parser.add_argument(
        "--unk_token",
        type=int,
        default=10000,
        help=( "vocab size")
    )
    args = parser.parse_args()
