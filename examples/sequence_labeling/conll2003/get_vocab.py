embedding_path = './data/glove.6B.100d.txt'
vocab_path = './data/embedding_vocab.txt'

vocabs = set()
with open(embedding_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        sp_line = line.split()
        if len(sp_line)<5:
            continue
        vocabs.add(sp_line[0].strip())

for i in range(1, 20):
    if '0'*i not in vocabs:
        vocabs.add('0'*i)
        print(f"{'0'*i} is added to the vocab to prevent len(number)=={i} number oov" )
with open(vocab_path, 'w', encoding='utf-8') as f:
    f.writelines("\n".join(vocabs))
