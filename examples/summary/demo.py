# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dlk.demo import Demo

demo = Demo(
    display_config={"@display@summary": {}},
    process_config="./config/processor.jsonc",
    fit_config="./config/fit.jsonc",
    checkpoint="./logs/0/checkpoint/epoch=2-step=3.ckpt",
    scrolling=False,
)

# After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .
