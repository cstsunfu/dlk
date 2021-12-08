from dlkit.utils.logger import setting_logger
import pytorch_lightning as pl
# setting_logger("./benchmark.log")
setting_logger("./title.log")
from dlkit.train import Train
import json

pl.seed_everything(88)

# trainer = Train('./tasks/test_cls.hjson')
# trainer = Train('./examples/sequence_labeling/benchmark/crf_lstm_main.hjson')
# trainer = Train('./examples/sequence_labeling/benchmark/pretrain_main.hjson')
# trainer = Train('./examples/sequence_labeling/pretrained_ner/main.hjson')
trainer = Train('./examples/sequence_labeling/simple_ner/crf_lstm_main.hjson')
# print(json.dumps(trainer.configs[0], indent=4))
trainer.run()
