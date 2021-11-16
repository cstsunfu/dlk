import pickle as pkl

data = pkl.load(open('./processed_data.pkl', 'rb'))
# data = pkl.load(open('./meta.pkl', 'rb'))

print(len(data['token_embedding']))
print(data['token_embedding'][:10])
# print(data)

# class A(object):
    # """docstring for A"""
    # def __init__(self, a, b):
        # super(A, self).__init__()
        # self.a = a
        # self.b = b

    # def fun(self):
        # """TODO: Docstring for fun.
        # :returns: TODO

        # """
        # self.c = 5
        # return self.a

    # def load(self, attr):
        # """TODO: Docstring for load.
        # :returns: TODO

        # """
        # pass



# a = A(1, 2)
# print(a.__dict__)
# b = A(3, 4)
# b.__dict__ = a.__dict__

# print(b.fun())
