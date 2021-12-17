from dlk.utils.logger import setting_logger
import pytorch_lightning as pl
# setting_logger("./benchmark.log")
# setting_logger("./conll_norm_lstm_crf.log")
setting_logger("./conll_norm_char_lstm_crf.log")
from dlk.predict import Predict
import json

pl.seed_everything(88)

# trainer = Train('./examples/sequence_labeling/benchmark/pretrained/first_piece_lstm_crf_main.hjson')
predictor = Predict('./examples/sequence_labeling/conll2003/norm_char_lstm_crf/output/task=crf_lstm_lr=0.01_optimizer=sgd_dropout=0.5_batch_size=10_lstm_output_size=200/config.json', './examples/sequence_labeling/conll2003/norm_char_lstm_crf/output/task=crf_lstm_lr=0.01_optimizer=sgd_dropout=0.5_batch_size=10_lstm_output_size=200/default/checkpoints/epoch=0-step=1404.ckpt')
# trainer = Train('./examples/sequence_labeling/conll2003/norm_lstm_crf/main.hjson')
predictor.predict()

# trainer.run()
