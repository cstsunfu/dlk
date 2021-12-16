

```hjson
datamodule: {
    "_name": "base",
    "config": {
        "padding": 0,
        "pin_memory": false,
        "shuffle": {
            "train": true,
            "predict": false,
            "valid": false,
            "test": false,
            "online": false
        },
        "key_type_pair": [('x', 'float'), ('y', 'int')],
        "batch_size": 32,
    }
},
```


```hjson
datamodule@train: {
    "_name": "base",
    "dataloader@train": {
        "_base": "base",
        "collate": {
            "_name": "base",
            "config": {
                "padding": 0,
            }
        },
        "config": {
            'pin_memory': false, //if use cuda, this should set true
            'shuffle': true,
        }
    },
    "dataset@train": {
        "_name": "base",
        "config": {
            "key_type_pair": [('x', 'float'), ('y', 'int')],
        }
    },
},
datamodule@predict:{
    "_name": "predict_base",
    "dataloader": {
        "_base": "base",
        "collate": {
            "_base": "base",
            "config": {
                "padding":0
            }
        },
        "config": {
            'pin_memory': false, //if use cuda, this should set true
            'shuffle': false,
        }
    },
    "dataset": {
        "_name": "base",
        "config": {
            "key_type_pair": [('x', 'float'), ('y', 'int')],
        }
    },
},

{
    "_name": "base",
    "dataloader@train": {
        "_base": "base",
        "collate": {
            "_name": "base",
            "config": {
                "padding": 0,
            }
        },
        "config": {
            'pin_memory': false, //if use cuda, this should set true
            'shuffle': true,
        }
    },
    "dataset@train": {
        "_name": "base",
        "config": {
            "key_type_pair": [('x', 'float'), ('y', 'int')],
        }
    },
}
```
