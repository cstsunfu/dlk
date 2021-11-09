import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import pytorch_lightning as pl
from typing import Dict, List, Union
from torch.nn.utils.rnn import pad_sequence
TODO: train dataloader shuffle=True
    other shuffle=False

class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(16, 64)
        self.decoder = nn.Linear(64, 2)
        self.all_data = {}

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = x['x']
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch['x'], batch['y']
        y = y.squeeze()
        print(batch['_index'])
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.cross_entropy(x_hat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss,on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        """TODO: Docstring for test_step.

        :batch: TODO
        :batch_idx: TODO
        :returns: TODO

        """
        x, y = batch['x'], batch['y']
        y = y.squeeze()
        # print(batch['_index'])
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.cross_entropy(x_hat, y)
        # Logging to TensorBoard by default
        # self.log("train_loss", loss)
        # print(f"test_step loss: {loss}")
        return loss

    def test_epoch_end(self, outputs):
        """TODO: Docstring for test_epoch_end.
        :returns: TODO

        """
        # print(f"test_epoch_end outputs: {outputs}")
        # self.log("result", outputs)
        print(self.train_data)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



class CustomDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data

        self.get = [('x', 'float'), ('y', 'int')]
        self.type_map = {'float': torch.float, 'int': torch.long, 'bool': torch.bool}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        one_ins = {}
        for key, key_type in self.get:
            one_ins[key] = torch.tensor(self.data.iloc[idx][key], dtype=self.type_map[key_type])
        one_ins['_index'] = torch.tensor([idx],  dtype=torch.long)
        return one_ins

# class SimpleCustomBatch:
    # def __init__(self, data):
        # # print(data)
        # # self.data = data
        # print(len(data))
        # print(data[0])
        # # transposed_data = list(zip(*data))
        # # self.inp = torch.stack(transposed_data[0], 0)
        # # self.tgt = torch.stack(transposed_data[1], 0)

    # # custom memory pinning method on custom type
    # def pin_memory(self):
        # self.inp = self.inp.pin_memory()
        # self.tgt = self.tgt.pin_memory()
        # return self


def collate_wrapper(batch):
    keys = batch[0].keys()
    data_map: Dict[str, Union[List[torch.Tensor], torch.Tensor]] = {}
    for key in keys:
        data_map[key] = []
    for key in keys:
        for one_ins in batch:
            data_map[key].append(one_ins[key])
    for key in data_map:
        data_map[key] = pad_sequence(data_map[key], batch_first=True, padding_value=0)
        # torch.cat([ ], pad=0)
    return data_map


np.random.seed(42)
inp = [np.random.randn(np.random.randint(15, 17)) for _ in range(20)]
# print(inp)
target = list(np.random.randint(0, 1, (20, 1)))
# print(target)

label = ["label"]*20

data = pd.DataFrame(data={"x": inp, 'y': target, "label": label})
dataset = CustomDataset(data)
train_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_wrapper, pin_memory=False)

# for data in train_loader:
    # print(data)
    # break
    # pass
# # init model
autoencoder = LitAutoEncoder()
autoencoder.train_data = data

# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
# trainer = pl.Trainer(gpus=8) (if you have GPUs)
trainer = pl.Trainer(profiler="simple", max_steps=2)
# trainer.test(model=autoencoder)
# print(trainer.predict(model=autoencoder, dataloaders=train_loader))
print(trainer.test(autoencoder, dataloaders=train_loader))
