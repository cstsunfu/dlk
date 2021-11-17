import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, dataloader, random_split, Dataset
import pandas as pd
import time
import pytorch_lightning as pl
from typing import Dict, List, Union
from torch.nn.utils.rnn import pad_sequence
# TODO: train dataloader shuffle=True
    # other shuffle=False

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, dataset, collate_fn):
        super().__init__()
        self.dataset = dataset
        self.collate_fn = collate_fn

    # def train_dataloader(self):
        # return DataLoader(self.mnist_train, batch_size=self.batch_size)

    # def val_dataloader(self):
        # return DataLoader(self.mnist_val, batch_size=self.batch_size)

    # def test_dataloader(self):
        # # return DataLoader(self.mnist_test, batch_size=self.batch_size)
        # self.train_dataloader = lambda self: DataLoader(self.dataset, batch_size=4, collate_fn=self.collate_fn, pin_memory=False)

    def test_dataloader(self):
        """TODO: Docstring for predict_dataloader.
        :returns: TODO

        """
        return DataLoader(self.dataset, batch_size=4, collate_fn=self.collate_fn, pin_memory=False, num_workers=4)
        # return None

    def val_dataloader(self):
        """TODO: Docstring for predict_dataloader.
        :returns: TODO

        """
        # return DataLoader(self.dataset, batch_size=4, collate_fn=self.collate_fn, pin_memory=False)
        return DataLoader(self.dataset, batch_size=4, collate_fn=self.collate_fn, pin_memory=False, num_workers=4)
        # return None

    def predict_dataloader(self):
        """TODO: Docstring for predict_dataloader.
        :returns: TODO

        """
        return DataLoader(self.dataset, batch_size=4, collate_fn=self.collate_fn, pin_memory=False)

    def train_dataloader(self):
        """TODO: Docstring for predict_dataloader.
        :returns: TODO

        """
        return DataLoader(self.dataset, batch_size=4, collate_fn=self.collate_fn, pin_memory=False, num_workers=4)
        # return DataLoader([])


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(16, 64)
        self.decoder = nn.Linear(64, 2)
        self.all_data = {}

    def forward(self, x: Dict[str, torch.Tensor]):
        # in lightning, forward defines the prediction/inference actions
        x = x['x']
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch['x'], batch['y']
        # y = y.squeeze()
        # print(batch['_index'])
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.cross_entropy(x_hat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss,on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """TODO: Docstring for test_step.

        :batch: TODO
        :batch_idx: TODO
        :returns: TODO

        """
        x, y = batch['x'], batch['y']
        # print(batch['_index'])
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.cross_entropy(x_hat, y)
        # Logging to TensorBoard by default
        self.log("validation_loss", loss)
        print(f"validation_step loss: {loss}")
        return loss

    def test_step(self, batch, batch_idx):
        """TODO: Docstring for test_step.

        :batch: TODO
        :batch_idx: TODO
        :returns: TODO

        """
        x, y = batch['x'], batch['y']
        # print(batch['_index'])
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.cross_entropy(x_hat, y)
        # Logging to TensorBoard by default
        # self.log("train_loss", loss)
        # print(f"test_step loss: {loss}")
        # if self.local_rank == 0:
            # print("start sleep")
            # time.sleep(1)
            # print("end sleep")
        return {"loss": loss, "index": batch['_index'], "x_hat": x_hat, 'y': y}

    def test_epoch_end(self, outputs):
        """TODO: Docstring for test_epoch_end.
        :returns: TODO

        """
        # self.log("result", outputs)
        gather_output = self.all_gather(outputs)
        if self.local_rank==0:
            # print(f"test_epoch_end outputs:", outputs)
            print(f"All gather :", gather_output)
            # for key in gather_output:
                # print(f"{key}\t: {gather_output[key].shape}")
        # self.log('------------------------------', rank_zero_only=True)
        # mean = torch.mean(self.all_gather(outputs))
        # print(f"Mean: {mean}")

        # # When logging only on rank 0, don't forget to add
        # # ``rank_zero_only=True`` to avoid deadlocks on synchronization.
        # if self.trainer.is_global_zero:
            # self.log("my_reduced_metric", mean, rank_zero_only=True)
        # print(self.train_data)
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
        try:
            data_map[key] = pad_sequence(data_map[key], batch_first=True, padding_value=0)
        except:
            if data_map[key][0].size():
                raise ValueError(f"The {data_map[key]} can not be concat by pad_sequence.")
            _data = pad_sequence([i.unsqueeze(0) for i in data_map[key]], batch_first=True, padding_value=0).squeeze()
            if not _data.size():
                _data.unsqueeze_(0)
            data_map[key] = _data
            # print( pad_sequence([i.unsqueeze(0) for i in data_map[key]], batch_first=True, padding_value=0).size()).
        # torch.cat([ ], pad=0)
    return data_map

if __name__ == "__main__":
    np.random.seed(20)
    inp = [np.random.randn(np.random.randint(16, 17)) for _ in range(20)]
    # print(inp)
    target = list(np.random.randint(0, 1, (20)))
    # print(target)

    label = ["label"]*20

    data = pd.DataFrame(data={"x": inp, 'y': target, "label": label})
    dataset = CustomDataset(data)

    # train_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_wrapper, pin_memory=False)

    data_module = MNISTDataModule(dataset, collate_wrapper)
    # class MNISTDataModule(pl.LightningDataModule):
        # def __init__(self, dataset, collate_fn):
    # for data in train_loader:
        # print(data)
        # break
        # pass
    # # init model
    autoencoder = LitAutoEncoder()
    autoencoder.train_data = data_module.dataset

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    # trainer = pl.Trainer(profiler="simple", max_steps=100, val_check_interval=0.5, log_every_n_steps=10)
    trainer = pl.Trainer(max_steps=20, val_check_interval=0.5, log_every_n_steps=2, accelerator="cpu", strategy="ddp", num_processes=2)
    # trainer = pl.Trainer(max_steps=20, val_check_interval=0.5, log_every_n_steps=2)

    # trainer.test(model=autoencoder)
    # trainer.fit(model=autoencoder, datamodule=data_module)
    # # print(trainer.predict(model=autoencoder, dataloaders=data_module))
    # autoencoder.load_state_dict(torch.load('./test.pt'))
    # # print('test ')
    # autoencoder.to_torchscript("script.sc")
    print("Print", trainer.test(autoencoder, datamodule=data_module))
    # torch.save(autoencoder.state_dict(), './test.pt')
