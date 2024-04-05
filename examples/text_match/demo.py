# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dlk.demo import Demo

demo = Demo(
    display_config={"@display@txt_match": {}},
    process_config="./config/processor.jsonc",
    fit_config="./config/fit.jsonc",
    checkpoint="./logs/0/checkpoint/epoch=8-step=36.ckpt",
)

# The boy is playing on the swings after school.
# A little boy in a gray and white striped sweater and tan pants is playing on a piece of playground equipment.
