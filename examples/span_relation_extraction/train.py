# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import lightning as pl

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


pl.seed_everything(88)

trainer = Train("./config/fit.jsonc")

trainer.run()
