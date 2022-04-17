# Data download

## For Bert based Model
From [BERT-base](https://huggingface.co/bert-base-cased/tree/main) download `tokenizer.json`, `config.json` and `pytorch_model.bin`, then save to `./data/bert/`.

# Do prepro use the prepro.hjson config

# Do train use the main.hjson config

# [Optional] Convert the result to bio with json2bio.py

# [Optional] Eval using conlleval.pl script

Note: Convert the result is because using conlleval script will get few diffrente result( $\Delta$<0.5%).
