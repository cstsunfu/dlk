from dlkit.utils.logger import setting_logger
setting_logger("./test.log")

from dlkit.train import Train

# trainer = Train('./tasks/test_cls.hjson')
trainer = Train('./examples/sequence_labeling/simple_ner/crf_lstm_main.hjson')
# trainer = Train('./tasks/test_crf_token_cls.hjson')
# trainer = Train('./tasks/config.json')
trainer.run()
