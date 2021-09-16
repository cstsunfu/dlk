import sys
sys.path.append('..')
from utils.logger import setting_logger
from utils.config import Config

setting_logger("./logger.log")

config = {
    "a":  {
        "a1": "a11",
        "a2": "a21"
    },
    "b": {
        "b1": "b11",
        "b2": "b21"
    }
}

config2 = {
    "a":  {
        "a2": "a21"
    },
    "b": {
        "b2": "b21"
    }
}

class TestConfig(Config):
    """docstring for TestConfig"""
    def __init__(self, **kwargs):
        super(TestConfig, self).__init__(**kwargs)
        self.a = kwargs.get('a')
        self.c = kwargs.pop("c", None)
        
    def no_imp(self):
        """TODO: Docstring for no_imp.
        :returns: TODO

        """
        return TestConfig(**config2)



conf = TestConfig(**config)
print(conf.to_dict())
print(conf.no_imp().to_dict())
# a = {}
# a['config'] = TestConfig
# print(a['config'](**config).a)

