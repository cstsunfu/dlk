import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import lightning as pl

import dlk.ray_optuna
from dlk import register
from dlk.train import Train


@register("additional_loss_collect", "user")
def loss_sum(losses, **args):
    """
    Args:
        losses: loss with key
    Returns:
        sum of losses
    """
    rt_config = args["rt_config"]
    loss = 0
    for key in losses:
        loss += losses[key]
        if key == "loss@cross_entropy#entity":
            loss += losses[key]
        else:
            loss += losses[key] * min(rt_config["current_epoch"] / 10, 2)
    return loss


if __name__ == "__main__":
    pl.seed_everything(88)

    trainer = Train("./config/optuna.jsonc")

    trainer.run()
