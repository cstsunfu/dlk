from dlk.utils.logger import setting_logger
import pytorch_lightning as pl
# setting_logger("./benchmark.log")
# setting_logger("./conll_norm_lstm_crf.log")
setting_logger("./conll_norm_char_lstm_crf.log")
from dlk.train import Train
import json

pl.seed_everything(88)

# trainer = Train('./examples/sequence_labeling/benchmark/pretrained/first_piece_lstm_crf_main.hjson')
trainer = Train('./examples/sequence_labeling/conll2003/norm_char_lstm_crf/main.hjson')
# trainer = Train('./examples/sequence_labeling/conll2003/norm_lstm_crf/main.hjson')

trainer.run()
