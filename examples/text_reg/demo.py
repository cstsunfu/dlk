# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dlk.demo import Demo

demo = Demo(
    display_config={"@display@txt_reg": {}},
    process_config="./config/processor.jsonc",
    fit_config="./config/fit.jsonc",
    checkpoint="./logs/0/checkpoint/epoch=6-step=28.ckpt",
)

#
