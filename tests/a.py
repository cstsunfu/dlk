
from typing import Callable, List, Dict, Tuple, Union
class A(object):
    """docstring for A"""
    def __init__(self, arg):
        super(A, self).__init__()
        self.arg = arg
        print("self.arg {}".format(arg))
        if isinstance(self.arg, dict):
            print(self.arg)
            for item in self.arg:
                self.get_base_config(self.arg[item])
                self.out()

    def out(self):
        """TODO: Docstring for out.
        :returns: TODO

        """
        print('out')

    @classmethod
    def get_base_config(cls, arg):
        """TODO: Docstring for get_base_config.

        :arg1: TODO
        :returns: TODO

        """
        return cls(arg)

source = 'a.b.c'
to = '1.2.3'
# source_list = source.split('.')
# to_list = to.split('.')
# for s, t in zip(source_list, to_list):
    # print(s, t)

a = {1:3}
# a[1] = 3
print(a)
# config = {
    # "a":{"b":{
        # "c":3
    # }},
    # "1":{"2":{
        # "3":8
    # }},
# }
# def link_para(link: Dict={}):
    # """TODO: Docstring for link_para.

    # :link: TODO
    # :returns: TODO

    # """
    # if not link:
        # return
    # source_config = config
    # to_config = config
    # for (source, to) in link.items():
        # source_list = source.split('.')
        # to_list = to.split('.')
        # for s, t in zip(source_list[:-1], to_list[:-1]):
            # source_config = source_config[s]
            # to_config = to_config[t]
        # to_config[to_list[-1]] = source_config[source_list[-1]]

# link_para({source:to})
# print(config)
