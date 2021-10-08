## Config Example

```json
    {
        "process": [
            {
                "_name": "space_tokenizer"
                "_status": ["train", "predict", "online"],
                "config": {
                    "data_pair": {
                        "origin": "origin_tokens"
                    }, // string or list, to_data[input['data'][data_set[..]]][to_data]=fn(input['data'][data_set[..]][from_data])
                    "data_set": ['train', 'dev'],
                    "map_char_token_idx": "origin_char_token_idx_map" // if this is empty string, will not do this
                }
            }, //0 the process num
            {
                "_name": "token_gather"
                "_status": ["train", "predict", "online"],
                "config": {
                    "tokens": "origin", // string or list, gather all filed
                    "deliver": "all_tokens",
                    "data_set": ['train', 'dev']
                }
            }, //1
            {
                "_name": "token_gather",
                "_status": ["train"],
                "config":{
                    "tokens": "label",
                    "deliver": "all_labels",
                    "data_set": ['train', 'dev'],
                    "deliver_method": "create_map"
                }
            }, //2
            {
                "_name": "label_to_id",
                "_status": ["train", "predict", "online"],
                "config": {
                    "data_pair": {
                        "label": "label_id"
                    }
                    "from_data": ["label"],
                    "to_data": ["label_id"],
                    "data_set": ['train', 'dev'],
                }
            }, //3
            {
                "_name": "get_static_embedding",
                "_status": ["train"],
                "config": {
                    "all_tokens": $ref,
                    "restructure_embedding_id": true
                    "deliver": "embedding",
                    "deliver_method": "new"
                }
            }, //4

            {
                "_name": "token_to_id",
                "_status": ["train", "predict", "online"],
                "config": {
                    "token_ids": $ref,
                    "token": "origin_tokens",
                    "ids": "origin_token_ids",
                    "data_set": ['train', 'dev'],
                }
            }, //5
        ],

        "save": {
            "train": { // status, use "train.predict" means "train" & "predict"
                "data.train": "./train.pkl",
                "data.dev": "./dev.pkl",
                "token_ids": "./token_ids.pkl",
                "embedding": "./embedding.pkl",
                "label_ids": "./label_ids.pkl",
            },
            "predict": {
                "data.predict": "./predict.pkl"
            }
        },
        
        "load": {
            "predict.online":{
                "token_ids": "./token_ids.pkl",
                "embedding": "./embedding.pkl",
                "label_ids": "./label_ids.pkl",
            }
        },
        "_link": {
            "process.0.config.save_tokens":"process.2.config.tokens",
            "process.2.config.token_ids":"process.-1.config.token_ids",
        }
    }
```

## To Process Data Format Example

```json
{
    "data": {
        "train": pd.DataFrame, // may include these columns "uuid"、"origin"、"label"
        "dev": pd.DataFrame, // may include these columns "uuid"、"origin"、"label"
    }
}

```

## Processed Data Format Example

```json
{
    "data": {
        "train": pd.DataFrame, // may include these columns "uuid"、"origin"、"labels"、"origin_tokens"、"label_ids"、"origin_token_ids"
        "dev": pd.DataFrame, // may include these columns "uuid"、"origin"、"labels"、"origin_tokens"、"label_ids"、"origin_token_ids"
    },
    "all_tokens": [..],
    "embedding": [..],
    "token_id_map": {"token": "id"},
    "id_token_map": {"id": "token"},
    "id_label_map": {"id": "label"},
    "label_id_map": {"label": "id"}
}
```
