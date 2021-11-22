import pickle as pkl

data = pkl.load(open('./processed_data.pkl', 'rb'))

for k in data:
    print(k)

print(data['label_vocab'])
print(data['data'])
# # data = pkl.load(open('./meta.pkl', 'rb'))

# print(len(data['token_embedding']))
# print(data['token_embedding'][:10])
# # print(data)
