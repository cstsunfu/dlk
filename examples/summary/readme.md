### Text Summary Example

![Summary]("../../pics/summary.png")

#### Dataset

We only provide some examples, you can add your own dataset like the format

#### how to run

0. Prepare the pretrained bart model `facebook/bart-large-cnn`, and run the `scripts/convert_hf_bart.py` convert the model to `dlk` format

1. Preprocess the data

Update the path to tokenizer `tokenizer.json` field at `config/processor.jsonc`(both encoder tokenizer and decoder tokenizer)
```
python process.py
```

2. Train the model

Update the path to pretrained bart `pretrained_model_path` field at `config/fit.jsonc`
```
python train.py
```
