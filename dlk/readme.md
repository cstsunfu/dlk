# Appointments

## Data format
### Input
For one sentence processor:

The input one sentence named "sentence", label named "labels"

The output named:
```
    "input_ids",
    "label_ids",
    "word_ids",
    "attention_mask",
    "special_tokens_mask",
    "type_ids", 
    "sequence_ids",
    "char_ids",
```


The input two sentence named "sentence_a", "sentence_b", label named "labels"

The output named:
```
    "input_ids",
    "label_ids",
    "word_ids",
    "attention_mask",
    "special_tokens_mask",
    "type_ids", 
    "sequence_ids",
    "char_ids",
```
### MASK
We set mask==1 for used data, mask==0 for useless data

### Batch First
All data set batch_first=True
    
## Model appointments

* All dropout put on output or intern of the module, no dropout for the module input


# The main file tree:

```
.
├── train.py-------------------------: train entry 
├── predict.py-----------------------: predict entry
├── process.py-----------------------: process entry
├── online.py------------------------: online entry
├── managers-------------------------: pytorch_lightning or other trainer
│   └── lightning.py-----------------: 
├── configures-----------------------: all default or specifical config
│   ├── core-------------------------: 
│   │   ├── callbacks----------------: 
│   │   ├── imodels------------------: 
│   │   ├── layers-------------------: 
│   │   │   ├── decoders-------------: 
│   │   │   ├── embeddings-----------: 
│   │   │   └── encoders-------------: 
│   │   ├── losses-------------------: 
│   │   ├── models-------------------: 
│   │   ├── modules------------------: 
│   │   └── optimizers---------------: 
│   ├── data-------------------------: 
│   │   ├── datamodules--------------: 
│   │   ├── processors---------------: 
│   │   └── subprocessors------------: 
│   ├── managers---------------------: 
│   └── tasks------------------------: 
├── core-----------------------------: *core* pytorch or other model code
│   ├── base_module.py---------------: base module for "layers"
│   ├── callbacks--------------------: 
│   ├── imodels----------------------: 
│   ├── layers-----------------------: 
│   │   ├── decoders-----------------: 
│   │   ├── embeddings---------------: 
│   │   └── encoders-----------------: 
│   ├── losses-----------------------: 
│   ├── models-----------------------: 
│   ├── modules----------------------: 
│   ├── optimizers-------------------: 
│   └── schedules--------------------: 
├── data-----------------------------: *core* code for data process or manager
│   ├── datamodules------------------: 
│   ├── postprocessors---------------: 
│   ├── processors-------------------: 
│   └── subprocessors----------------: 
└── utils----------------------------: 
    ├── config.py--------------------: process config(dict) toolkit
    ├── get_root.py------------------: get project root path
    ├── logger.py--------------------: logger
    ├── parser.py--------------------: parser config
    ├── register.py------------------: register the module to a registry
    ├── tokenizer_util.py------------: tokenizer util
    └── vocab.py---------------------: vocabulary
```


# Config Parser Rules

## Inherit

Simple e.g.

```

default.hjson
{
    _base:  parant,
    config: {
        "will_be_rewrite": 3     
    }
}

parant.hjson
{
    _name:  base_config,
    config: {
        "will_be_rewrite": 1,
        "keep": 8     
    }
}

You have the two config named default.hjson, and  parant.hjson, the parser result will be :
{
    _name:  base_config,
    config: {
        "will_be_rewrite": 3,
        "keep": 8     
    }
}
```

## Grid Search

Simple e.g.

```
{
    _name: search_example,
    config: {
        "para1": 10,
        "para2": 20,
        "para3": 30,
        _search: {
            "para1": [11, 12],
            "para2": [21, 22],
         }
    }
}

given the  above config, the parser result will be a list of possible configure which length is 4.
[
    {
        _name: search_example,
        config: {
            "para1": 11,
            "para2": 21,
            "para3": 30,
        }
    },
    {
        _name: search_example,
        config: {
            "para1": 12,
            "para2": 21,
            "para3": 30,
        }
    },
    {
        _name: search_example,
        config: {
            "para1": 11,
            "para2": 22,
            "para3": 30,
        }
    },
    {
        _name: search_example,
        config: {
            "para1": 12,
            "para2": 22,
            "para3": 30,
        }
    },
]
```

## Link(Argument Passing)

    Parameters are not allowed to be assigned repeatedly (the same parameter cannot appear more than once in the target position, otherwise it will cause ambiguity.)
    If a low level link  wer all not appeared at before, it will be directly regist them.
    If only one of key or value appeared in high level _links, the value of the key and value will be overwritten by the corresponding value of the upper level,
    If they both appeared at before, and if they linked the same value, we will do nothing, otherwise `RAISE AN ERROR`

```
Simple e.g.

child.hjson
{
    "_base": parant,
    "config": {
        "para1": 1,
        "para2": 2,
        "para3": 3,
    }
    "_link": {"config.para1": "config.para2"}
}

parant.hjson

{
    "_name": parant,
    "config":{
        "para1": 4,
        "para2": 5,
        "para3": 6,
    }
    "_link": {"config.para2": "config.para1"}
}

the 
result.hjson

{
    "_name": parant,
    "config":{
        "para1": 1,
        "para2": 1,  # call link({"config.para1": "config.para2"})
        "para3": 3,
    }
}
```

## Focus(Representation)

The focus part is for simple the logger file, we will use the value of focus dict to replace the key while logging.


## SubModule(Combination)

Due to we using the dict to represent a config, and the key is regarded as the submodule name, but sometimes one top level module will have two or more same submodules(with different config). 
You can set the submodule name as 'submodule@speciel_name'.
