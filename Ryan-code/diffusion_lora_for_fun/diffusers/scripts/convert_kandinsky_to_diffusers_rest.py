import argparse
import os
import tempfile

import torch
from accelerate import load_checkpoint_and_dispatch

from diffusers import UNet2DConditionModel
from diffusers.models.transformers.prior_transformer import PriorTransformer
from diffusers.models.vq_model import VQModel


"""
Example - From the diffusers root directory:

Download weights:
```sh
$ wget https://huggingface.co/ai-forever/Kandinsky_2.1/blob/main/prior_fp16.ckpt
```

Convert the model:
```sh
python scripts/convert_kandinsky_to_diffusers.py \
      --prior_checkpoint_path /home/yiyi_huggingface_co/Kandinsky-2/checkpoints_Kandinsky_2.1/prior_fp16.ckpt \
      --clip_stat_path  /home/yiyi_huggingface_co/Kandinsky-2/checkpoints_Kandinsky_2.1/ViT-L-14_stats.th \
      --text2img_checkpoint_path /home/yiyi_huggingface_co/Kandinsky-2/checkpoints_Kandinsky_2.1/decoder_fp16.ckpt \
      --inpaint_text2img_checkpoint_path /home/yiyi_huggingface_co/Kandinsky-2/checkpoints_Kandinsky_2.1/inpainting_fp16.ckpt \
      --movq_checkpoint_path /home/yiyi_huggingface_co/Kandinsky-2/checkpoints_Kandinsky_2.1/movq_final.ckpt \
      --dump_path /home/yiyi_huggingface_co/dump \
      --debug decoder
```
"""


# prior

PRIOR_ORIGINAL_PREFIX = "model"

# Uses default arguments
PRIOR_CONFIG = {}










# done prior

# unet

# We are hardcoding the model configuration for now. If we need to generalize to more model configurations, we can
# update then.

UNET_CONFIG = {
    "act_fn": "silu",
    "addition_embed_type": "text_image",
    "addition_embed_type_num_heads": 64,
    "attention_head_dim": 64,
    "block_out_channels": [384, 768, 1152, 1536],
    "center_input_sample": False,
    "class_embed_type": None,
    "class_embeddings_concat": False,
    "conv_in_kernel": 3,
    "conv_out_kernel": 3,
    "cross_attention_dim": 768,
    "cross_attention_norm": None,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
    ],
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "encoder_hid_dim": 1024,
    "encoder_hid_dim_type": "text_image_proj",
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 3,
    "mid_block_only_cross_attention": None,
    "mid_block_scale_factor": 1,
    "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "num_class_embeds": None,
    "only_cross_attention": False,
    "out_channels": 8,
    "projection_class_embeddings_input_dim": None,
    "resnet_out_scale_factor": 1.0,
    "resnet_skip_time_act": False,
    "resnet_time_scale_shift": "scale_shift",
    "sample_size": 64,
    "time_cond_proj_dim": None,
    "time_embedding_act_fn": None,
    "time_embedding_dim": None,
    "time_embedding_type": "positional",
    "timestep_post_act": None,
    "up_block_types": [
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "upcast_attention": False,
    "use_linear_projection": False,
}






# done unet

# inpaint unet

# We are hardcoding the model configuration for now. If we need to generalize to more model configurations, we can
# update then.

INPAINT_UNET_CONFIG = {
    "act_fn": "silu",
    "addition_embed_type": "text_image",
    "addition_embed_type_num_heads": 64,
    "attention_head_dim": 64,
    "block_out_channels": [384, 768, 1152, 1536],
    "center_input_sample": False,
    "class_embed_type": None,
    "class_embeddings_concat": None,
    "conv_in_kernel": 3,
    "conv_out_kernel": 3,
    "cross_attention_dim": 768,
    "cross_attention_norm": None,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
    ],
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "encoder_hid_dim": 1024,
    "encoder_hid_dim_type": "text_image_proj",
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 9,
    "layers_per_block": 3,
    "mid_block_only_cross_attention": None,
    "mid_block_scale_factor": 1,
    "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "num_class_embeds": None,
    "only_cross_attention": False,
    "out_channels": 8,
    "projection_class_embeddings_input_dim": None,
    "resnet_out_scale_factor": 1.0,
    "resnet_skip_time_act": False,
    "resnet_time_scale_shift": "scale_shift",
    "sample_size": 64,
    "time_cond_proj_dim": None,
    "time_embedding_act_fn": None,
    "time_embedding_dim": None,
    "time_embedding_type": "positional",
    "timestep_post_act": None,
    "up_block_types": [
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "upcast_attention": False,
    "use_linear_projection": False,
}






# done inpaint unet


# unet utils


# <original>.time_embed -> <diffusers>.time_embedding


# <original>.input_blocks.0 -> <diffusers>.conv_in






# <original>.out.0 -> <diffusers>.conv_norm_out


# <original>.out.2 -> <diffusers>.conv_out


# <original>.input_blocks -> <diffusers>.down_blocks


# <original>.middle_block -> <diffusers>.mid_block


# <original>.output_blocks -> <diffusers>.up_blocks






# TODO maybe document and/or can do more efficiently (build indices in for loop and extract once for each split?)


# done unet utils








# movq

MOVQ_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "down_block_types": ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "AttnDownEncoderBlock2D"),
    "up_block_types": ("AttnUpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
    "num_vq_embeddings": 16384,
    "block_out_channels": (128, 256, 256, 512),
    "vq_embed_dim": 4,
    "layers_per_block": 2,
    "norm_type": "spatial",
}






















if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    parser.add_argument(
        "--prior_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the prior checkpoint to convert.",
    )
    parser.add_argument(
        "--clip_stat_path",
        default=None,
        type=str,
        required=False,
        help="Path to the clip stats checkpoint to convert.",
    )
    parser.add_argument(
        "--text2img_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the text2img checkpoint to convert.",
    )
    parser.add_argument(
        "--movq_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the text2img checkpoint to convert.",
    )
    parser.add_argument(
        "--inpaint_text2img_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help="Path to the inpaint text2img checkpoint to convert.",
    )
    parser.add_argument(
        "--checkpoint_load_device",
        default="cpu",
        type=str,
        required=False,
        help="The device passed to `map_location` when loading checkpoints.",
    )

    parser.add_argument(
        "--debug",
        default=None,
        type=str,
        required=False,
        help="Only run a specific stage of the convert script. Used for debugging",
    )

    args = parser.parse_args()

    print(f"loading checkpoints to {args.checkpoint_load_device}")

    checkpoint_map_location = torch.device(args.checkpoint_load_device)

    if args.debug is not None:
        print(f"debug: only executing {args.debug}")

    if args.debug is None:
        print("to-do")
    elif args.debug == "prior":
        prior_model = prior(args=args, checkpoint_map_location=checkpoint_map_location)
        prior_model.save_pretrained(args.dump_path)
    elif args.debug == "text2img":
        unet_model = text2img(args=args, checkpoint_map_location=checkpoint_map_location)
        unet_model.save_pretrained(f"{args.dump_path}/unet")
    elif args.debug == "inpaint_text2img":
        inpaint_unet_model = inpaint_text2img(args=args, checkpoint_map_location=checkpoint_map_location)
        inpaint_unet_model.save_pretrained(f"{args.dump_path}/inpaint_unet")
    elif args.debug == "decoder":
        decoder = movq(args=args, checkpoint_map_location=checkpoint_map_location)
        decoder.save_pretrained(f"{args.dump_path}/decoder")
    else:
        raise ValueError(f"unknown debug value : {args.debug}")