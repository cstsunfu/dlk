import pandas as pd



class DataSet(object):
    """docstring for DataSet"""
    def __init__(self, config):
        super(DataSet, self).__init__()
        self.config = config
        

    def process_instance(self, origin_instance: dict):
        """TODO: Docstring for process.

        :origin_instance: TODO
        :returns: TODO

        """
        pass

    def process(self, data: pd.DataFrame)->pd.DataFrame:
        """TODO: Docstring for process.

        :data: TODO
        :returns: TODO

        """
        pass

        return data
