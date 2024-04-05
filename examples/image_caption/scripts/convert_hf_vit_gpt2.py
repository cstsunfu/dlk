import argparse
import json
import os

import torch
from tokenizers import Tokenizer
from transformers import VisionEncoderDecoderModel

encoder_prefix_pair = [
    "model.encoder.pretrained_transformers.model.encoder",
    "encoder.encoder",
]
decoder_prefix_pair = [
    "model.decoder.bart_like_decoder.gpt2_decoder",
    "decoder.transformer",
]

specific_map = {
    "model.embedding.weight": "decoder.transformer.wte.weight",
    "model.lm_head.weight": "decoder.lm_head.weight",
    "model.encoder.pretrained_transformers.model.embeddings.cls_token": "encoder.embeddings.cls_token",
    "model.encoder.pretrained_transformers.model.embeddings.position_embeddings": "encoder.embeddings.position_embeddings",
    "model.encoder.pretrained_transformers.model.embeddings.patch_embeddings.projection.weight": "encoder.embeddings.patch_embeddings.projection.weight",
    "model.encoder.pretrained_transformers.model.embeddings.patch_embeddings.projection.bias": "encoder.embeddings.patch_embeddings.projection.bias",
    "model.encoder.pretrained_transformers.model.layernorm.weight": "encoder.layernorm.weight",
    "model.encoder.pretrained_transformers.model.layernorm.bias": "encoder.layernorm.bias",
    "model.encoder.pretrained_transformers.model.pooler.dense.weight": "encoder.pooler.dense.weight",
    "model.encoder.pretrained_transformers.model.pooler.dense.bias": "encoder.pooler.dense.bias",
    "model.decoder.bart_like_decoder.gpt2_decoder.wte.weight": "decoder.transformer.wte.weight",
    "model.decoder.bart_like_decoder.gpt2_decoder.wpe.weight": "decoder.transformer.wpe.weight",
    "model.decoder.bart_like_decoder.gpt2_decoder.ln_f.weight": "decoder.transformer.ln_f.weight",
    "model.decoder.bart_like_decoder.gpt2_decoder.ln_f.bias": "decoder.transformer.ln_f.bias",
}

encoder_layer_weight = [
    "attention.attention.query.weight",
    "attention.attention.query.bias",
    "attention.attention.key.weight",
    "attention.attention.key.bias",
    "attention.attention.value.weight",
    "attention.attention.value.bias",
    "attention.output.dense.weight",
    "attention.output.dense.bias",
    "intermediate.dense.weight",
    "intermediate.dense.bias",
    "output.dense.weight",
    "output.dense.bias",
    "layernorm_before.weight",
    "layernorm_before.bias",
    "layernorm_after.weight",
    "layernorm_after.bias",
]
decoder_layer_weight = [
    "ln_1.weight",
    "ln_1.bias",
    "attn.bias",
    "attn.masked_bias",
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "attn.c_proj.bias",
    "ln_2.weight",
    "ln_2.bias",
    "crossattention.bias",
    "crossattention.masked_bias",
    "crossattention.c_attn.weight",
    "crossattention.c_attn.bias",
    "crossattention.q_attn.weight",
    "crossattention.q_attn.bias",
    "crossattention.c_proj.weight",
    "crossattention.c_proj.bias",
    "ln_cross_attn.weight",
    "ln_cross_attn.bias",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    "mlp.c_proj.bias",
]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_path", type=str, default="../pretrain/")
    parser.add_argument("--layer_num", type=int, default=12)

    args = parser.parse_args()
    vit_config_path = os.path.join(args.pretrain_path, "vit")
    gpt2_config_path = os.path.join(args.pretrain_path, "gpt2")
    os.makedirs(vit_config_path, exist_ok=True)
    os.makedirs(gpt2_config_path, exist_ok=True)

    with open(os.path.join(args.pretrain_path, "config.json"), "r") as f:
        config = json.load(f)
    with open(os.path.join(vit_config_path, "config.json"), "w") as f:
        json.dump(config["encoder"], f, indent=4)

    with open(os.path.join(gpt2_config_path, "config.json"), "w") as f:
        json.dump(config["decoder"], f, indent=4)

    with open(os.path.join(args.pretrain_path, "tokenizer.json"), "r") as f:
        tokenizer = json.load(f)
    with open(os.path.join(gpt2_config_path, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f, indent=4)

    with open(os.path.join(args.pretrain_path, "preprocessor_config.json"), "r") as f:
        preprocessor_config = json.load(f)
    with open(os.path.join(vit_config_path, "preprocessor_config.json"), "w") as f:
        json.dump(preprocessor_config, f, indent=4)

    tensors = torch.load(
        os.path.join(args.pretrain_path, "pytorch_model.bin"), map_location="cpu"
    )

    dlk_name_map = {}
    for layer in range(args.layer_num):
        for encoder_weight_name in encoder_layer_weight:
            dlk_name_map[
                f"{encoder_prefix_pair[0]}.layer.{layer}.{encoder_weight_name}"
            ] = f"{encoder_prefix_pair[1]}.layer.{layer}.{encoder_weight_name}"
        for decoder_weight_name in decoder_layer_weight:
            dlk_name_map[
                f"{decoder_prefix_pair[0]}.h.{layer}.{decoder_weight_name}"
            ] = f"{decoder_prefix_pair[1]}.h.{layer}.{decoder_weight_name}"
        for k, v in specific_map.items():
            dlk_name_map[k] = v
    dlk_checkpoint = {"state_dict": {}}
    for k, v in dlk_name_map.items():
        dlk_checkpoint["state_dict"][k] = tensors[v]

    torch.save(dlk_checkpoint, os.path.join(args.pretrain_path, "dlk_model.bin"))
