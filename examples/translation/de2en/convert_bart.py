import torch
import numpy as np

model = torch.load('./pytorch_model.bin')

dlk_dict = {}
for i in model:
    name_map = {
            "encoder": "encoder.bart_encoder.bart_encoder",
            "decoder": "decoder.bart_decoder.bart_decoder",
            "shared": ["source_embedding.embedding", "target_embedding.embedding"],
            "lm_head.": "lm_head",
    }

    ignore_set = {
            "encoder.embed_tokens.weight",
            "decoder.embed_tokens.weight"
    }
    flag = False
    if i in ignore_set:
        print(f"Ignore the weights {i}")
        continue
    for origin_prefix, new_prefixs in name_map.items():
        if i.startswith(origin_prefix):
            flag = True
            if isinstance(new_prefixs, list):
                for new_prefix in new_prefixs:
                    dlk_dict[new_prefix+str(i[len(origin_prefix):])] = model[i]
                    print(f"Convert {i:60} -> {new_prefix+str(i[len(origin_prefix):])}")
            else:
                new_prefix = new_prefixs
                dlk_dict[new_prefix+str(i[len(origin_prefix):])] = model[i]
                print(f"Convert {i:60} -> {new_prefix+str(i[len(origin_prefix):])}")
    if not flag:
        print(f"Cannot convert {i}")

torch.save(dlk_dict, "dlk_model.bin")
