#!/usr/bin/env python3
import argparse
import fnmatch

from safetensors.torch import load_file

from diffusers import Kandinsky3UNet


MAPPING = {
    "to_time_embed.1": "time_embedding.linear_1",
    "to_time_embed.3": "time_embedding.linear_2",
    "in_layer": "conv_in",
    "out_layer.0": "conv_norm_out",
    "out_layer.2": "conv_out",
    "down_samples": "down_blocks",
    "up_samples": "up_blocks",
    "projection_lin": "encoder_hid_proj.projection_linear",
    "projection_ln": "encoder_hid_proj.projection_norm",
    "feature_pooling": "add_time_condition",
    "to_query": "to_q",
    "to_key": "to_k",
    "to_value": "to_v",
    "output_layer": "to_out.0",
    "self_attention_block": "attentions.0",
}

DYNAMIC_MAP = {
    "resnet_attn_blocks.*.0": "resnets_in.*",
    "resnet_attn_blocks.*.1": ("attentions.*", 1),
    "resnet_attn_blocks.*.2": "resnets_out.*",
}
# MAPPING = {}






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert U-Net PyTorch model to Kandinsky3UNet format")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the original U-Net PyTorch model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the converted model")

    args = parser.parse_args()
    main(args.model_path, args.output_path)