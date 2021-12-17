import hjson
from tokenizers import Tokenizer
from tokenizers.models import WordPiece, BPE
from tokenizers.trainers import WordPieceTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import argparse
from tokenizers import normalizers
from tokenizers import pre_tokenizers
import sys
sys.path.append('../')
from dlk.utils.tokenizer_util import TokenizerNormalizerFactory, PreTokenizerFactory, TokenizerPostprocessorFactory


def get_trainer(trainer_name:str):
    """TODO: Docstring for get_trainer.
    :returns: TODO

    """
    trainers = {
        "wordpiece": WordPieceTrainer,
        "bpe": BpeTrainer
    }
    if trainer_name not in trainers:
        raise KeyError(f"There is not a trainer named {trainer_name}")
    return trainers[trainer_name]


def get_model(model_name:str):
    models = {
        "wordpiece": WordPiece,
        "bpe": BPE,
    }
    if model_name not in models:
        raise KeyError(f"There is not a model named {model_name}")
    return models[model_name]


def get_processor(factory, one_processor):
    """TODO: Docstring for _get_processor.

    :factory: TODO
    :one_processor: TODO
    :returns: TODO

    """
    if isinstance(one_processor, dict):
        assert len(one_processor) == 1
        process_name, process_config = list(one_processor.items())[0]
        return factory.get(process_name)(**process_config)
    else:
        assert isinstance(one_processor, str)
        return factory.get(one_processor)()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='./train_tokenizer.hjson',
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
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default='wordpiece',
        choices=['wordpiece'],
        help=( "tokenizer type [wordpiece, bpe, etc.]")
    )
    parser.add_argument(
        "--special_tokens",
        type=list,
        default=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        help=( "tokenizer special tokens")
    )
    parser.add_argument(
        "--out_tokenizer",
        type=str,
        default="tokenizer.json",
        help=( "tokenizer output")
    )
    parser.add_argument(
        "--pre_tokenizer",
        type=list,
        default=["whitespace"],
        help=( "pretoeknizer the str")
    )
    parser.add_argument(
        "--post_processor",
        type=str,
        default='',
        help=( "post processor the str")
    )
    parser.add_argument(
        "--normalizer",
        type=list,
        default=[],
        help=( "normalizer the str before toeknizer")
    )
    args = parser.parse_args()

    if args.config:
        config_json = hjson.load(open(args.config), object_pairs_hook=dict)
        for key, value in config_json.items():
           setattr(args, key, value)

    # print(args)
    Trainer = get_trainer(args.tokenizer_type)
    Model = get_model(args.tokenizer_type)
    trainer = Trainer(vocab_size=args.vocab_size, special_tokens=args.special_tokens)

    tokenizer = Tokenizer(Model(unk_token=args.unk_token))

    pretokenizer_factory = PreTokenizerFactory()
    tokenizer_postprocessor_factory = TokenizerPostprocessorFactory()
    tokenizer_normalizer_factory = TokenizerNormalizerFactory()

    if args.pre_tokenizer:
        assert isinstance(args.pre_tokenizer, list)
        pre_tokenizers_list = []
        for one_pre_tokenizer in args.pre_tokenizer:
            pre_tokenizers_list.append(get_processor(pretokenizer_factory, one_pre_tokenizer))
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pre_tokenizers_list)
        # print(pre_tokenizers_list)

    if args.post_processor:
        assert isinstance(args.post_processor, str) or isinstance(args.post_processor, dict)
        tokenizer.post_processor = get_processor(tokenizer_postprocessor_factory, args.post_processor)

    if args.normalizer:
        assert isinstance(args.normalizer, list)
        normalizers_list = []
        for one_normalizer in args.normalizer:
            normalizers_list.append(get_processor(tokenizer_normalizer_factory, one_normalizer))
        tokenizer.normalizer = normalizers.Sequence(normalizers_list)

    tokenizer.train(args.train_files, trainer)
    tokenizer.save(args.out_tokenizer, pretty=True)
