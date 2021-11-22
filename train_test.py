from dlkit.utils.logger import setting_logger
setting_logger("./test.log")
from dlkit.train import Train


trainer = Train('./tasks/test_cls.hjson')
trainer.run()

