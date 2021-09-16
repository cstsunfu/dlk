## WIP

## What is this project?

* Provide a templete for deep learning (especially for nlp) training and deploy.
* Provide parameters search.
* Provide basic architecture search.
* Provide some basic modules and models.
* Provide basic deploy method.

## Config example

The root(system) config:

```json
{
    __focus__: {
        "multi_models.model_1.decoder.config.output_size": "decoder.output_size=",
        "multi_models.model_2.encoder": "encoder="
    },
    multi_models: {
        __name__: "distill",
        model_1: {
            __link__: {
                "encoder.config.output_size": "decoder.config.input_size"
            },
            __name__: "lstm_linear_ner",
            decoder: {
                config: {
                    output_size: 8
                }
            },
            encoder: {
                config: {
                    __search__: {
                        output_size: [
                            3,
                            5
                        ]
                    }
                }
            }
        },
        model_2: {
            __name__: "lstm_linear_ner",
            decoder: {
                config: {
                    output_size: 8
                }
            },
            encoder: {
                config: {
                    output_size: 8
                }
            }
        }
    }
}

```

The model "lstm_linear_ner" config:
```json
{
    __name__: "ner",
    decoder: {
        __base__: "linear",
        config: {
            input_size: 200,
            output_size: 3
        }
    },
    encoder: {
        __base__: "lstm"
    }
}
```

The encoder "lstm" config:

```json
{
    config: {
        bidirection: true,
        hidden_size: 200,
        input_size: 200,
        layers: 1,
        output_size: 200,
    },
    __name__: "lstm",
}

```


The decoder "linear" config:

```json
{
    config: {
        input_size: 256,
        output_size: 2
    },
    __name__: "linear",
}
```

After the system parser:

```python
class Train(object):
    """docstring for Train"""
    def __init__(self, config_file):
        super(Train, self).__init__()
        self.config_file = self.load_hjson_file(config_file)
        self.focus = self.config_file.pop('__focus__', {})
        parser = CONFIG_PARSER_REGISTRY['system'](self.config_file)
        self.configs = parser.parser()
        self.config_names = []
        for possible_config in self.configs:
            config_name = []
            for source, to in self.focus.items():
                config_point = possible_config
                trace = source.split('.')
                for t in trace:
                    config_point = config_point[t]
                config_name.append(to+str(config_point))
            self.config_names.append('_'.join(config_name))

        for config, name in zip(self.configs, self.config_names):
            print(json.dumps(config))
            print(name)
```

The output is:

```json
{
    "multi_models": {
        "model_2": {
            "encoder": {
                "config": {
                    "output_size": 8
                }
            },
            "decoder": {
                "config": {
                    "output_size": 8
                }
            },
            "__name__": "lstm_linear_ner"
        },
        "model_1": {
            "encoder": {
                "config": {
                    "output_size": 3
                }
            },
            "decoder": {
                "config": {
                    "output_size": 8,
                    "input_size": 3
                }
            },
            "__name__": "lstm_linear_ner"
        },
        "__name__": "distill"
    }
}
decoder.output_size=8_encoder={'config': {'output_size': 8}}

{
    "multi_models": {
        "model_2": {
            "encoder": {
                "config": {
                    "output_size": 8
                }
            },
            "decoder": {
                "config": {
                    "output_size": 8
                }
            },
            "__name__": "lstm_linear_ner"
        },
        "model_1": {
            "encoder": {
                "config": {
                    "output_size": 5
                }
            },
            "decoder": {
                "config": {
                    "output_size": 8,
                    "input_size": 5
                }
            },
            "__name__": "lstm_linear_ner"
        },
        "__name__": "distill"
    }
}
decoder.output_size=8_encoder={'config': {'output_size': 8}}

```
