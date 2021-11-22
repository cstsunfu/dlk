import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec
import hjson
import argparse
# Data from https://pytorch.org/text/_modules/torchtext/datasets/language_modeling.html


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

    print("Preprocessing text...")
    lines = []
    for file in args.train_files:
        lines = lines + list(LineSentence(file))
    print("Training embedding...")

    model = Word2Vec(lines, vector_size=args.embedding_size, window=args.window, min_count=args.min_count, workers=args.workers, epochs=args.epochs, sg=args.sg, compute_loss=True, callbacks=[callback()])
    print("Saving embedding...")
    model.save(args.model_file)
    model.wv.save_word2vec_format(args.embedding_file, binary=False)
