import pandas as pd


data = {"a": ["1 f","2","3","4"], 'b': ["5","6","7","8"]}


df = pd.DataFrame(data=data)

d = df[['a', 'b']].values
# d = [i.values for i in d]
# b = df['b']


print(d)
