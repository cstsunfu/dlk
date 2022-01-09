# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
