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
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import logging
pl_logger = logging.getLogger("ligtning")
pl_logger.setLevel(logging.INFO)
# pl_logger.propagate = False
pl_logger.info("--------------------------------------------------------------------------------------------")


# loger.info(...)
# loger.debug(...)

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
    def val_dataloader(self):
        """TODO: Docstring for predict_dataloader.
        :returns: TODO

        """
        return DataLoader(self.dataset, batch_size=4, collate_fn=self.collate_fn, pin_memory=False, num_workers=4)

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
        self.decoder = nn.Linear(64, 7)
        self.other = nn.Linear(8, 8)
        self.all_data = {}

    def forward(self, x: Dict[str, torch.Tensor]):
        # in lightning, forward defines the prediction/inference actions
        x = x['x']
        embedding = self.encoder(x)
        return embedding

    def predict_step(self, batch, batch_idx):
        embedding = self(batch)
        return {"embedding": embedding, "index": batch["_index"]}

    def predict_step_end(self, outputs):
        """TODO: Docstring for predict_epoch_end.
        :returns: TODO

        """
        # print("outputs: ", outputs)

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
        return {"loss": loss.unsqueeze(0), "index": batch['_index'], "x_hat": x_hat, 'y': y}

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
        return {"loss": loss.unsqueeze(0), "index": batch['_index'], "x_hat": x_hat, 'y': y}

    def validation_epoch_end(self, outputs):
        """TODO: Docstring for test_epoch_end.
        :returns: TODO

        """
        print('batch num')
        print(len(outputs))
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
        # print(f"origin outputs on rank {self.local_rank}: {outputs}")
        def proc_dist_outputs(dist_outputs):
            """gather all distributed outputs to outputs which is like in a single worker.

            :dist_outputs: the inputs of pytorch_lightning *_epoch_end when using ddp
            :returns: the inputs of pytorch_lightning *_epoch_end when only run on one worker.
            """
            outputs = []
            for dist_output in dist_outputs:
                one_output = {}
                for key in dist_output:
                    try:
                        one_output[key] = torch.cat(torch.swapaxes(dist_output[key], 0, 1).unbind(), dim=0)
                    except:
                        raise KeyError(f"{key}: {dist_output[key]}")
                outputs.append(one_output)
            return outputs
        if self.trainer.world_size>1:
            dist_outputs = self.all_gather(outputs)
            if self.local_rank in [0, -1]:
                outputs = proc_dist_outputs(dist_outputs)

        if self.local_rank in [0, -1]:
            key_all_batch_map = {}
            for batch in outputs:
                for key in batch:
                    if key not in key_all_batch_map:
                        key_all_batch_map[key] = []
                    key_all_batch_map[key].append(batch[key])
            key_all_ins_map = {}
            for key in key_all_batch_map:
                key_all_ins_map[key] = torch.cat(key_all_batch_map[key], dim=0)

            # print(key_all_ins_map)
        return outputs


    def test_epoch_end(self, outputs):
        """TODO: Docstring for test_epoch_end.
        :returns: TODO

        """
        print('batch num')
        print(len(outputs))
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
        # print(f"origin outputs on rank {self.local_rank}: {outputs}")
        def proc_dist_outputs(dist_outputs):
            """gather all distributed outputs to outputs which is like in a single worker.

            :dist_outputs: the inputs of pytorch_lightning *_epoch_end when using ddp
            :returns: the inputs of pytorch_lightning *_epoch_end when only run on one worker.
            """
            outputs = []
            for dist_output in dist_outputs:
                one_output = {}
                for key in dist_output:
                    try:
                        one_output[key] = torch.cat(torch.swapaxes(dist_output[key], 0, 1).unbind(), dim=0)
                    except:
                        raise KeyError(f"{key}: {dist_output[key]}")
                outputs.append(one_output)
            return outputs
        if self.trainer.world_size>1:
            dist_outputs = self.all_gather(outputs)
            if self.local_rank in [0, -1]:
                outputs = proc_dist_outputs(dist_outputs)

        if self.local_rank in [0, -1]:
            key_all_batch_map = {}
            for batch in outputs:
                for key in batch:
                    if key not in key_all_batch_map:
                        key_all_batch_map[key] = []
                    key_all_batch_map[key].append(batch[key])
            key_all_ins_map = {}
            for key in key_all_batch_map:
                key_all_ins_map[key] = torch.cat(key_all_batch_map[key], dim=0)

            # print(key_all_ins_map)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # print("configure optim")
        # for group in optimizer.param_groups:
            # # print(f"optimizer group： {group}")
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
        one_ins['_index'] = torch.tensor(idx,  dtype=torch.long)
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
    # pl.seed_everything(seed=21, workers=False)
    # loger = logging.getLogger('lightning')

    pl.seed_everything( workers=False)
    inp = [np.random.randn(np.random.randint(16, 17)) for _ in range(8)]
    # print(inp)
    target = [np.random.randint(0, 7) for _ in range(8)]
    # print(target)

    label = ["label"+str(i) for i in target]

    data = pd.DataFrame(data={"x": inp, 'y': target, "label": label})
    dataset = CustomDataset(data)

    # train_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper, pin_memory=False)

    data_module = MNISTDataModule(dataset, collate_wrapper)
    # class MNISTDataModule(pl.LightningDataModule):
        # def __init__(self, dataset, collate_fn):
    # for data in train_loader:
        # print(data)
        # break
        # pass
    # # init model
    autoencoder = LitAutoEncoder()
    # optimizer = optim.AdamW(params=[
        # { "params": [p for p in autoencoder.encoder.parameters()], "lr": 0.3},
        # { "params": autoencoder.decoder.parameters(), "lr": 0.3},
        # { "params": autoencoder.other.parameters()},
    # ], lr=0.1
    # )
    # print(optimizer)
    # print(list(autoencoder.parameters()))







    autoencoder.train_data = data_module.dataset

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    # trainer = pl.Trainer(profiler="simple", max_steps=100, val_check_interval=0.5, log_every_n_steps=10)
    logger = TensorBoardLogger(save_dir="tb_logger", version="", name="", sub_dir='test_sub_dir_2')
    # trainer = pl.Trainer(max_steps=20, val_check_interval=0.5, log_every_n_steps=2, accelerator="cpu", strategy="ddp", num_processes=3, default_root_dir='root_dir', logger=logger)

    trainer = pl.Trainer(max_steps=20, val_check_interval=0.5, log_every_n_steps=2, logger=logger, default_root_dir='test_sub_dir_2', callbacks=[ModelCheckpoint(dirpath='checkpoint_dirpath', filename='checkpoint_filename')])

# # default logger used by trainer
    # # trainer = pl.Trainer(max_steps=20, val_check_interval=0.5, log_every_n_steps=2)
    # # trainer.test(model=autoencoder)
    # trainer.fit(model=autoencoder, datamodule=data_module)
    # # print(trainer.predict(model=autoencoder, dataloaders=data_module))
    # # print(trainer.world_size)
    # # trainer.predict(model=autoencoder, dataloaders=data_module)
    # # autoencoder.load_state_dict(torch.load('./test.pt'))
    # # # print('test ')
    # # autoencoder.to_torchscript("script.sc")
    trainer.validate(autoencoder, datamodule=data_module)
    # # torch.save(autoencoder.state_dict(), './test.pt')
