import lightning as pl
from dlk.train import Train

if __name__ == "__main__":
    pl.seed_everything(88)
    # 1. Initialize and Run Trainer
    trainer = Train("./config/fit.jsonc")
    trainer.run()
    print("Training finished. Checkpoint saved to ./logs/0/checkpoint/")
