import torch
import torch.nn as nn
normal =  torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([2.0/10]))
# normal.sample
# print(normal.sample((20, 10)).shape)
a = nn.Embedding.from_pretrained(normal.sample((10, 5)).squeeze_(-1), padding_idx=-1)



b = torch.LongTensor([-1, 2])
print(a(b))
# print(a)
