### Span Classification Example(NER)

![Span Classification](../../pics/seq_lab.png)

#### Dataset

Test on the `conll2003` dataset.


#### how to run

1. Preprocess the data

Update the path to tokenizer `tokenizer_path` field at `config/processor.jsonc`
```
python process.py
```

2. Train the model

Update the path to pretrained bert/distilbert `pretrained_model_path` field at `config/fit.jsonc`
```
python train.py
```



### Result


```
Default
efficent=False

relation_position=False

Epoch 7/7  ──────────────────────────────────────── 439/439 0:00:30 _ 0:00:00 14.49it/s train_loss: 0.076 val_precision: 90.672 val_recall: 91.041 val_f1: 90.856 val_valid_loss: 0.63
```

```
efficent=True
Epoch 7/7  ──────────────────────────────────────── 439/439 0:00:29 _ 0:00:00 14.70it/s train_loss: 0.004 val_precision: 91.145 val_recall: 91.856 val_f1: 91.499 val_valid_loss: 0.500

```

```
relation_position=True
Epoch 7/7  ──────────────────────────────────────── 439/439 0:00:30 _ 0:00:00 14.38it/s train_loss: 0.038 val_precision: 91.677 val_recall: 92.050 val_f1: 91.863 val_valid_loss: 0.703
```
