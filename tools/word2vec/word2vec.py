import multiprocessing
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import hjson
import argparse
# Data from https://pytorch.org/text/_modules/torchtext/datasets/language_modeling.html
from gensim.models.word2vec_inner import MAX_WORDS_IN_BATCH

def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert `text` (bytestring in given encoding or unicode) to unicode.

    Parameters
    ----------
    text : str
        Input text.
    errors : str, optional
        Error handling behaviour if `text` is a bytestring.
    encoding : str, optional
        Encoding of `text` if it is a bytestring.

    Returns
    -------
    str
        Unicode version of `text`.

    """
    if isinstance(text, str):
        return text
    return str(text, encoding, errors=errors)

class Sentences(object):
    def __init__(self, file_names, max_sentence_length=MAX_WORDS_IN_BATCH):
        self.file_names = file_names
        self.max_sentence_length = max_sentence_length

    def __iter__(self):
        for fname in self.file_names:
            print(f"Training on {fname}")
            for line in open(fname, 'r'):
                line = any2unicode(line).split()
                i = 0
                while i < len(line):
                    yield line[i: i + self.max_sentence_length]
                    i += self.max_sentence_length

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='./word2vec.hjson',
        help="The config path.",
    )
    parser.add_argument(
        "--train_files", type=list, default=['train.txt'], help="train tokenizer files."
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=128,
        help=( "vector size")
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=10,
        help=( "minist word frequence")
    )
    parser.add_argument(
        "--max_vocab_size",
        type=int,
        default=300000,
        help=( "max vocab size")
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=( "0 means use all cpus")
    )
    parser.add_argument(
        "--window",
        type=int,
        default=6,
        help=( "window size")
    )
    parser.add_argument(
        "--sg",
        type=bool,
        default=0,
        help=( "use skip gram or not. if set to 0 use cbow")
    )
    parser.add_argument(
        "--hs",
        type=bool,
        default=1,
        help=( "If 1, hierarchical softmax will be used for model training.  If 0, and `negative` is non-zero, negative sampling will be used.")
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help=( "training epochs")
    )
    parser.add_argument(
        "--embedding_file",
        type=str,
        default='embedding.txt',
        help=( "embedding file")
    )
    args = parser.parse_args()
    if args.config:
        config_json = hjson.load(open(args.config), object_pairs_hook=dict)
        for key, value in config_json.items():
           setattr(args, key, value)
    if not args.workers:
        args.workers = multiprocessing.cpu_count()
    # print(args)

    print("Start training...")
    lines = Sentences(args.train_files)

    model = Word2Vec(lines, vector_size=args.embedding_size, hs=args.hs, window=args.window, min_count=args.min_count, workers=args.workers, epochs=args.epochs, sg=args.sg, compute_loss=True, max_vocab_size=args.max_vocab_size, callbacks=[callback()])
    print("Saving embedding...")
    model.save(args.model_file)
    model.wv.save_word2vec_format(args.embedding_file, binary=False)
