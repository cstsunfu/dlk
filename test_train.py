from dlkit.utils.logger import setting_logger
import pytorch_lightning as pl
# setting_logger("./benchmark.log")
setting_logger("./title.log")
from dlkit.train import Train
import json

pl.seed_everything(88)

# trainer = Train('./examples/sequence_labeling/benchmark/pretrained/first_piece_lstm_crf_main.hjson')
trainer = Train('./examples/sequence_labeling/conll2003/norm_char_lstm_crf/main.hjson')
# print(json.dumps(trainer.configs[0], indent=4))
trainer.run()
