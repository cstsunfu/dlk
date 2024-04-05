# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import lightning as pl

from dlk.train import Train

pl.seed_everything(88)

trainer = Train("./config/fit.jsonc")

trainer.run()
