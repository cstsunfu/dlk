class A(object):
    """docstring for A"""
    def __init__(self, param_a):
        self.param_a = param_a
    
    def function(self):
        """TODO: Docstring for function_from_a.
        :returns: TODO

        """
        print('function_from a')
    
    def func_a(self):
        """TODO: Docstring for function_from_a.
        :returns: TODO

        """
        self.function()


class B(object):
    """docstring for A"""
    def __init__(self, a):
        self.a = a
    
    def function(self):
        """TODO: Docstring for function_from_a.
        :returns: TODO

        """
        print('function_from B')
    


class C(B, A):
    """docstring for C"""
    def __init__(self, arg):
        A.__init__(self, param_a="nihao")
        B.__init__(self, a='test')
        self.arg = arg
        
c = C('a')
c.func_a()
