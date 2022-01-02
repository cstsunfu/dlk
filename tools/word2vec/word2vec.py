# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import hjson
import time
import argparse
import numpy as np
# Data from https://pytorch.org/text/_modules/torchtext/datasets/language_modeling.html
from gensim.models.word2vec_inner import MAX_WORDS_IN_BATCH
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)


def any2unicode(text, encoding: str='utf8', errors: str='strict'):
    """Convert `text` (bytestring in given encoding or unicode) to unicode.

    Args:
        text : Input text.
        errors : str, optional
            Error handling behaviour if `text` is a bytestring.
        encoding : str, optional
            Encoding of `text` if it is a bytestring.

    Returns:
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
        """read all the lines in the list of self.file_names, and break the sentence by the self.max_sentence_length
        Returns:
            Iterable strs

        """

        for fname in self.file_names:
            print(f"On {fname}")
            with open(fname, 'r') as fin:
                for line in fin.readlines():
                    line = any2unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i: i + self.max_sentence_length]
                        i += self.max_sentence_length


class LossCallback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1
        self.losses = []
        self.cumu_loss = 0.0
        self.previous_epoch_time = time.time()

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        norms = [np.linalg.norm(v) for v in model.wv.vectors]
        now = time.time()
        epoch_seconds = now - self.previous_epoch_time
        self.previous_epoch_time = now
        self.cumu_loss += float(loss)
        print(f"Loss after epoch {self.epoch}: {loss} (cumulative loss so far: {self.cumu_loss}) "+\
              f"-> epoch took {round(epoch_seconds, 2)} s - vector norms min/avg/max: "+\
              f"{round(float(min(norms)), 2)}, {round(float(sum(norms)/len(norms)), 2)}, {round(float(max(norms)), 2)}")

        self.epoch += 1
        self.losses.append(float(loss))
        # loss will overflow when the loss very large. see https://github.com/RaRe-Technologies/gensim/issues/2735
        model.running_training_loss = 0.0


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
        help=( "0/none means use all cpus")
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
        default=0,
        help=( "If 1, hierarchical softmax will be used for model training.  If 0, and `negative` is non-zero, negative sampling will be used.")
    )
    parser.add_argument(
        "--negative",
        type=int,
        default=10,
        help=( "Anti with hs==1, negative sampling num, num_ns (number of negative samples per positive context word) between [5, 20] is shown to work best for smaller datasets, while num_ns between [2,5] suffices for larger datasets.")
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
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

    model = Word2Vec(lines, vector_size=args.embedding_size, hs=args.hs, negative=args.negative, window=args.window, min_count=args.min_count, workers=args.workers, epochs=args.epochs, sg=args.sg, compute_loss=True, max_vocab_size=args.max_vocab_size, callbacks=[LossCallback()])
    print("Saving embedding...")
    model.save(args.model_file)
    model.wv.save_word2vec_format(args.embedding_file, binary=False)
