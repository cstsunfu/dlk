# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dlk.demo import Demo

demo = Demo(
    display_config={"@display@seq_lab": {"render_height": 400, "render_width": 600}},
    process_config="./config/bert_firstpiece_lstm_crf/processor.jsonc",
    fit_config="./config/bert_firstpiece_lstm_crf/fit.jsonc",
    checkpoint="./logs/optimizer=adamw/lr=8e-05/checkpoint/epoch=7-step=32.ckpt",
)

# After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .
