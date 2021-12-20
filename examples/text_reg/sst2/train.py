from dlk.utils.logger import Logger
import pytorch_lightning as pl
from dlk.train import Train
import json

logger = Logger('log.txt')
pl.seed_everything(88)

# trainer = Train('./examples/sequence_labeling/benchmark/pretrained/first_piece_lstm_crf_main.hjson')
trainer = Train('./distil_bert/main.hjson')
# print(json.dumps(trainer.configs, indent=4))

trainer.run()
