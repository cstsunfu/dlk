class A(object):
    """docstring for A"""
    def __init__(self, a):
        super(A, self).__init__()
        self.a = a
        
a = {"b":3, "a": A(3)}


def fun(a, b):
    """TODO: Docstring for fun.

    :t: TODO
    :returns: TODO

    """
    print(a)

fun(**a)
