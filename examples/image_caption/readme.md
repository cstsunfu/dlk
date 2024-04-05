### Image Caption Example

![Image Caption]("../../pics/img_cap.png")

#### Dataset

Test on the `phiyodr/coco2017` dataset. We just use the pretrained model, so we only provide some examples


#### how to run

0. Download the model `nlpconnect/vit-gpt2-image-captioning` and run `./scripts/convert_hf_vit_gpt2.py` convert the `hf` model to `dlk` model

1. Preprocess the data

Update the path to tokenizer `tokenizer_path` and image preprocess `preprocess_config` fields at `config/processor.jsonc`
```
python process.py
```

2. Train the model

Update the path to pretrained vit `pretrained_model_path` and gpt2 `pretrained_model_path` field at `config/fit.jsonc`
```
python train.py # for train we reuse the pretrained model, so in the code there is a point to the converted model
```
