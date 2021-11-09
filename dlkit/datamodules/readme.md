'''
dataset
{
    "_name": "base",
    "config": {
        "key_type_pair": [('x', 'float'), ('y', 'int')],
        'pin_memory': false //if use cuda, this should set true
    }
}

'''


```hjson
{
    "_name": "base",
    "dataloader@train": {
        "_base": "base",
        "collate": {
            "_name": "base",
            "config": {
                "padding": 0,
            }
        }

        "config": {
            'pin_memory': false //if use cuda, this should set true
        }
    },
    "dataset@train": {
        "_name": "base",
        "config": {
            "key_type_pair": [('x', 'float'), ('y', 'int')],
        }
    },
    "config": {
        "datasets": {
            "train": {
                "data": "train"
            },
        }
    }
}

```
