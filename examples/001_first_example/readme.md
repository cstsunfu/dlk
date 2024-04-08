### The First Project

This example is the same as `../text_cls` but from scratch.

#### Prepare the BERT model and .intc.json

Download the BERT pretrained model and the `tokenizer.json` to `./pretrain`

add a file `.intc.json` in current dir
```json
{
    "entry": [
        "config" // the config will put in this dir, the intc-lsp will server on files in this dir
    ],
    "src": [ // the source code for this project
        "dlk", // the dlk package
        "src" // and the src code will write
    ]
}

```

#### Prepare the Dataset

load and convert the `sst2` dataset, see `./process.py`

```python
from datasets import load_dataset

def flat(data):
    """flat the data like zip"""
    sentences = data["sentence"]
    uuids = data["uuid"]
    labelses = data["labels"]
    return [
        {"sentence": sentece, "labels": [labels], "uuid": uuid}
        for sentece, labels, uuid in zip(sentences, labelses, uuids)
    ]


label_map = {0: "neg", 1: "pos"}

data = load_dataset("sst2")   # load dataset from huggingface dataset or what you want
data = data.map(
    lambda one: {
        "sentence": one["sentence"],
        "labels": label_map[one["label"]],
        "uuid": str(uuid.uuid1()),
    },
    remove_columns=["sentence", "label"],
)
input = {
    "train": pd.DataFrame(flat(data["train"].to_dict())),       # convert the train data and valid part to pd.DataFrame
    "valid": pd.DataFrame(flat(data["validation"].to_dict())),
}
```

#### Prepare the Process code

dlk provide a general `@processor@default` processor to preprocess the data, but for this example we create a simple one, please check the code on `./src/my_process.py` and registed as `@processor@my`

#### Prepare the config of preprocessor
config `@processor@my` see `./config/processor.jsonc`,

#### Run the preprocessor

then run preprocess

```bash
python process.py
```

#### Prepare the model code

dlk provide many modules, but we want just test ours.

we define a simple classification model at `./src/my_model.py` and registed as `@model@my_model`

#### Prepare the DataModule
dlk also provide a general datamodule, for this case, we implement ours.

we define a simple datamodule at `./src/my_datamodule.py` and registed as `@datamodule@my_datamodule`

#### Prepare the `fit.json`

besides `datamodule` `model`, there are many other module like `loss`, `optimizer`, `schedule` which is easy to understand, we reuse them buitin dlk.

see `./config/fit.jsonc`


#### Train your model

just prepare the train.py and
```bash
python train.py

```
