import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CyclicLR
import copy

class Encoder(nn.Module):
    """docstring for Base"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.elinear1 = nn.Linear(3, 4)
        self.elinear2 = nn.Linear(4, 3)
        

class Base(nn.Module):
    """docstring for Base"""
    def __init__(self):
        super(Base, self).__init__()
        self.encoder = Encoder()

        self.decoder = Encoder()
        # self.conv = nn.Conv1d(4, 6, 3)
        self.linear = nn.Linear(4, 6)


        
model = Base()


all_named_parameters = list(model.named_parameters())
has_grouped_params = set()

optimizer_special_groups = [["bias & LayerNorm.bias & LayerNorm.weight", {"weight_decay": 0}]]

params = []
for i, special_group in enumerate(optimizer_special_groups):
    assert len(special_group) == 2
    key, group_config = copy.deepcopy(special_group)
    keys = [s.strip() for s in key.split('&')]
    group_param = []
    for n, p  in all_named_parameters:
        if n in has_grouped_params:
            continue
        if any(key in n for key in keys):
            has_grouped_params.add(n)
            group_param.append(p)
    group_config['params'] = group_param
    # group_config['name'] = str(i)+' group'
    params.append(group_config)

reserve_params = [p for n, p in all_named_parameters if not n in has_grouped_params]
# params.append({"params": reserve_params, "name": "default"})
params.append({"params": []})


optimizer = optim.AdamW(params=params, lr=1e-3)

print(len(optimizer.param_groups))
# print(params)
