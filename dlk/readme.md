
# TODO

- [ ] all_gather_object
- [ ] pre-trained model init without weight



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


### Task naming appointments

DLK处理的所有问题我们都看做一个任务，而一个任务又会划分为多个子任务, 子任务又可以有自己的子任务，下面是一个任务的定义方式:

```
{
    "_name": "task_name", //or "_base", "base_task_name"
    "_link": {}, // this is reserved keywords
    "_search: {}, // this is reserved keywords"
    "sub_task1":{
    },
    "sub_task2":{
    }
}

```

由于所有的任务他们本身又可以被视为其他任务的子任务，所以我们就来看一下关于一个子任务的一些约定
```
这是一个子任务的配置格式

{
    "sub_task_name": {
        "_name": "sub_task_config_name",
        ...config
    }
}

or

{
    "sub_task_name": {
        "_base": "base_sub_task_config_name",
        ...additional config
    }
}

```
配置的key表示这个子任务

`sub_task_name` 的命名一般会表示该子任务在这个task中所扮演的角色，而每个子任务一般都是由dlk的一个专门的模块进行处理，比如`processor`任务中的`subprocessor`子任务均由`dlk.data.subprocessors`这个模块集合(这个里面会有多个subprocessor)进行处理，为了能区分不同的`subprocessor`我们在对`sub_task_name`进行命名时会采用`subprocessor@subprocessor_name_a`来表明我们使用的是`subprocessors`这个模块集合中的具有`subprocessor_name_a`这个功能的`subprocessor`来处理.

对于配置文件中的 `_base` 或 `_name` 模块的命名则会省略掉key中已经包含的`sub_task_name`

采用 `AA@BB#CC`的方式对一个子任务的configure进行命名

其中 `AA`表示处理`sub_task_name`所在表示的模块集合中的具体模块名，比如最常见的`basic`表示使用`basic`模块处理这个子任务，处理方法在对应模块集合中的名为`basic`中定义的逻辑处理

`BB`表明这个config处理的是什么问题比如（`seq_lab`/`txt_cls`/ets.）, `CC`则表明处理这个问题的配置文件的核心特征
    
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
