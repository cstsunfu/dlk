import argparse
import json
import os

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

encoder_prefix_pair = ["model.encoder.bart_like_encoder.bart_encoder", "model.encoder"]
decoder_prefix_pair = ["model.decoder.bart_like_decoder.bart_decoder", "model.decoder"]

specific_map = {
    "model.decoder.bart_like_decoder.bart_decoder.embed_positions.weight": "model.decoder.embed_positions.weight",
    "model.decoder.bart_like_decoder.bart_decoder.embed_tokens.weight": "model.shared.weight",
    "model.decoder.bart_like_decoder.bart_decoder.layernorm_embedding.bias": "model.decoder.layernorm_embedding.bias",
    "model.decoder.bart_like_decoder.bart_decoder.layernorm_embedding.weight": "model.decoder.layernorm_embedding.weight",
    "model.lm_head.bias": "final_logits_bias",
    "model.lm_head.weight": "model.shared.weight",
    "model.embedding.weight": "model.shared.weight",
    "model.encoder.bart_like_encoder.bart_encoder.embed_tokens.weight": "model.shared.weight",
    "model.encoder.bart_like_encoder.bart_encoder.embed_positions.weight": "model.encoder.embed_positions.weight",
    "model.encoder.bart_like_encoder.bart_encoder.layernorm_embedding.bias": "model.encoder.layernorm_embedding.bias",
    "model.encoder.bart_like_encoder.bart_encoder.layernorm_embedding.weight": "model.encoder.layernorm_embedding.weight",
}

encoder_layer_weight = [
    "fc1.bias",
    "fc1.weight",
    "fc2.bias",
    "fc2.weight",
    "final_layer_norm.bias",
    "final_layer_norm.weight",
    "self_attn.k_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.out_proj.bias",
    "self_attn.out_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn_layer_norm.bias",
    "self_attn_layer_norm.weight",
]
decoder_layer_weight = [
    "encoder_attn.k_proj.bias",
    "encoder_attn.k_proj.weight",
    "encoder_attn.out_proj.bias",
    "encoder_attn.out_proj.weight",
    "encoder_attn.q_proj.bias",
    "encoder_attn.q_proj.weight",
    "encoder_attn.v_proj.bias",
    "encoder_attn.v_proj.weight",
    "encoder_attn_layer_norm.bias",
    "encoder_attn_layer_norm.weight",
    "fc1.bias",
    "fc1.weight",
    "fc2.bias",
    "fc2.weight",
    "final_layer_norm.bias",
    "final_layer_norm.weight",
    "self_attn.k_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.out_proj.bias",
    "self_attn.out_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn_layer_norm.bias",
    "self_attn_layer_norm.weight",
]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="../pretrain/model.safetensors_back"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="../pretrain/tokenizer.json"
    )
    parser.add_argument(
        "--model_config_path", type=str, default="../pretrain/config.json"
    )
    parser.add_argument("--layer_num", type=int, default=12)
    parser.add_argument("--output_path", type=str, default="../pretrain/dlk_model.bin")

    args = parser.parse_args()

    # if tokenizer._tokenizer.get_vocab_size()

    tensors = {}
    with safe_open(args.model_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k == "final_logits_bias":
                tensors[k] = f.get_tensor(k).squeeze()
            else:
                tensors[k] = f.get_tensor(k)

    real_vocab_size = Tokenizer.from_file(args.tokenizer_path).get_vocab_size()
    embed_vocab_size = json.load(open(args.model_config_path))["vocab_size"]
    if real_vocab_size != embed_vocab_size:
        print(
            f"vocab size mismatch: {real_vocab_size} vs {embed_vocab_size}, padding the embedding"
        )
        assert real_vocab_size > embed_vocab_size
        emb_weight = tensors["model.shared.weight"]
        emb_bias = tensors["final_logits_bias"]
        tensors["model.shared.weight"] = torch.cat(
            [emb_weight, emb_weight[embed_vocab_size - real_vocab_size :, :]], dim=0
        )
        tensors["final_logits_bias"] = torch.cat(
            [emb_bias, emb_bias[embed_vocab_size - real_vocab_size :]], dim=0
        )

    dlk_name_map = {}
    for layer in range(args.layer_num):
        for encoder_weight_name in encoder_layer_weight:
            dlk_name_map[
                f"{encoder_prefix_pair[0]}.layers.{layer}.{encoder_weight_name}"
            ] = f"{encoder_prefix_pair[1]}.layers.{layer}.{encoder_weight_name}"
        for decoder_weight_name in decoder_layer_weight:
            dlk_name_map[
                f"{decoder_prefix_pair[0]}.layers.{layer}.{decoder_weight_name}"
            ] = f"{decoder_prefix_pair[1]}.layers.{layer}.{decoder_weight_name}"
        for k, v in specific_map.items():
            dlk_name_map[k] = v
    dlk_checkpoint = {"state_dict": {}}
    for k, v in dlk_name_map.items():
        dlk_checkpoint["state_dict"][k] = tensors[v]

    torch.save(dlk_checkpoint, args.output_path)
