# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dlk import register
from dlk.demo import Demo

demo = Demo(
    display_config={
        "@display@span_relation": {
            # "input": {
            #     "sentence": "text",
            # }
        }
    },
    process_config="./config/processor.jsonc",
    fit_config="./config/fit.jsonc",
    checkpoint="./logs/0/checkpoint/epoch=28-step=2117.ckpt",
)
# ever since hamas won palestinian legislative elections last january , president bush and prime minister ehud olmert of israel have done everything they could think of to isolate hamas and far less than they might have to help fatah 's most important remaining leader , the palestinian president , mahmoud abbas .
