
outputs = []

outputs_map = {}
for batch in outputs:
    for key in batch:
        outputs_map[key] = outputs_map.get(key, []).append(batch[key])
    
    
