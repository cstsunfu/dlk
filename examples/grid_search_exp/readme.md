
### Grid Search on Text Match Classification Example

![Grid Search HP](../../pics/grid_search_hp.png)

#### Dataset

Test on the `snli` dataset.


#### how to run

1. Preprocess the data(same as the original task)

Update the path to tokenizer `tokenizer_path` field at `config/processor.jsonc`
```
python process.py
```

2. Train the model

Update the path to pretrained bert/distilbert `pretrained_model_path` field at `config/fit.jsonc`

Add the `_search` paras for what you want to search. In this example, we will get totally 8 specific configs

```
python train.py
```
