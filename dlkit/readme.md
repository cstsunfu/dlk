
# The main file tree:
```
├── train.py-------------------------: train entry 
├── predict.py-----------------------: predict entry
├── process.py-----------------------: process entry
├── online.py------------------------: online entry
├── callbacks------------------------: callbacks
│   ├── __init__.py------------------:
│   ├── checkpoint.py----------------:
│   ├── early_stop.py----------------:
│   ├── lr_monitor.py----------------:
│   └── weight_average.py------------:
├── configures-----------------------: all modules config
│   ├── datamodules------------------:
│   ├── decoders---------------------:
│   ├── embeddings-------------------:
│   ├── encoders---------------------:
│   ├── imodels----------------------:
│   ├── managers---------------------:
│   ├── models-----------------------:
│   └── ....-------------------------:
├── datamodules----------------------: dataloader dataset, etc.
├── imodels--------------------------: intergration model, model+loss+optimizer+schedule etc.
├── layers---------------------------: specifical encoders, decoders, and embeddings
│   ├── decoders---------------------:
│   ├── embeddings-------------------:
│   └── encoders---------------------:
├── losses---------------------------:
├── managers-------------------------: pytorch_lightning or other trainer
│   └── lightning.py-----------------:
├── models---------------------------: models (process inputs to specific outputs)
├── modules--------------------------: basic module
│   ├── identity.py------------------:
│   ├── linear.py--------------------:
│   ├── lstm.py----------------------:
│   └── ...--------------------------:
├── optimizers-----------------------: optimizers
├── postprocessors-------------------: postprocessors for specific task(like ner, clisification, etc.)
├── processors-----------------------: use subprocessors to process the origin data to processed data(tokenizer, token2id, etc.)
│   ├── basic.py---------------------:
│   └── readme.md--------------------:
├── readme.md------------------------:
├── schedules------------------------: learning rate schedules
├── subprocessors--------------------: subprocessor for processor
│   ├── fast_tokenizer.py------------:
│   ├── load.py----------------------:
│   ├── save.py----------------------:
│   ├── token2id.py------------------:
│   ├── token_embedding.py-----------:
│   └── token_gather.py--------------:
└── utils----------------------------: some basic module or useful tool
    ├── base_module.py---------------: base module for "layers"
    ├── config.py--------------------: process config(dict) toolkit
    ├── get_root.py------------------: get project root path
    ├── logger.py--------------------: logger
    ├── parser.py--------------------: parser config
    ├── register.py------------------: register the module to a registry
    ├── tokenizer_util.py------------: tokenizer util
    └── vocab.py---------------------: vocabulary
```



