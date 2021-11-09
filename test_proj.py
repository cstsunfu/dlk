from typing import Dict


class TestProj(object):
    """docstring for TestProj"""
    def __init__(self, arg):
        super(TestProj, self).__init__()
        self.arg = arg
        
    def on_processor_before(self)->Dict:
        """TODO: Docstring for on_processor_before.
        :returns: TODO

        """
        pass

    def on_processor_after(self, data:Dict)->Dict:
        """TODO: Docstring for on_processor_after.

        :arg1: TODO
        :returns: TODO

        """
        return data

    def on_predict_after(self):
        """TODO: Docstring for on_predict_after.
        :returns: TODO

        """
        pass
