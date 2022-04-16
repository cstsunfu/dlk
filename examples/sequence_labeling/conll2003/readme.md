# Data download

Download [Conll2003](https://data.deepai.org/conll2003.zip) and unzip to datadir

## For Bert based Model
From [BERT-base](https://huggingface.co/bert-base-cased/tree/main) download `tokenizer.json`, `config.json` and `pytorch_model.bin`, then save to `./data/bert/`.

## For LSTM based Model
1. You should download `glove.6b.100d.txt` and save to `./data/glove/`
2. You should gather all vocab in glove, and use the tool in tools.convert_tokenizer.vocab2tokenizer.py convert your vocab to `tokenizer.json` and save `tokenizer.json` to `./data/glove`
 
# Convert to json

```
python ./bio2json.py 
```

# Do prepro use the prepro.hjson config

# Do train use the main.hjson config

# Convert the result to bio with json2bio.py

# Eval using conlleval.pl script

 This is because the dlkit evaluation is not very same as conlleval script result
