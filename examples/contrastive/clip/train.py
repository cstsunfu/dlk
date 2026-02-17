import lightning as pl

from dlk.train import Train

if __name__ == "__main__":
    pl.seed_everything(88)
    trainer = Train("./config/fit.jsonc")
    trainer.run()
