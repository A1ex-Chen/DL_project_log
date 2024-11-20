import argparse
import tempfile

import torch
from accelerate import load_checkpoint_and_dispatch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import UnCLIPPipeline, UNet2DConditionModel, UNet2DModel
from diffusers.models.transformers.prior_transformer import PriorTransformer
from diffusers.pipelines.unclip.text_proj import UnCLIPTextProjModel
from diffusers.schedulers.scheduling_unclip import UnCLIPScheduler


r"""
Example - From the diffusers root directory:

Download weights:
```sh
$ wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/efdf6206d8ed593961593dc029a8affa/decoder-ckpt-step%3D01000000-of-01000000.ckpt
$ wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/4226b831ae0279020d134281f3c31590/improved-sr-ckpt-step%3D1.2M.ckpt
$ wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/85626483eaca9f581e2a78d31ff905ca/prior-ckpt-step%3D01000000-of-01000000.ckpt
$ wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/0b62380a75e56f073e2844ab5199153d/ViT-L-14_stats.th
```

Convert the model:
```sh
$ python scripts/convert_kakao_brain_unclip_to_diffusers.py \
      --decoder_checkpoint_path ./decoder-ckpt-step\=01000000-of-01000000.ckpt \
      --super_res_unet_checkpoint_path ./improved-sr-ckpt-step\=1.2M.ckpt \
      --prior_checkpoint_path ./prior-ckpt-step\=01000000-of-01000000.ckpt \
      --clip_stat_path ./ViT-L-14_stats.th \
      --dump_path <path where to save model>
```
"""


# prior

PRIOR_ORIGINAL_PREFIX = "model"

# Uses default arguments
PRIOR_CONFIG = {}










# done prior


# decoder

DECODER_ORIGINAL_PREFIX = "model"

# We are hardcoding the model configuration for now. If we need to generalize to more model configurations, we can
# update then.
DECODER_CONFIG = {
    "sample_size": 64,
    "layers_per_block": 3,
    "down_block_types": (
        "ResnetDownsampleBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
        "SimpleCrossAttnDownBlock2D",
    ),
    "up_block_types": (
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "SimpleCrossAttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ),
    "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
    "block_out_channels": (320, 640, 960, 1280),
    "in_channels": 3,
    "out_channels": 6,
    "cross_attention_dim": 1536,
    "class_embed_type": "identity",
    "attention_head_dim": 64,
    "resnet_time_scale_shift": "scale_shift",
}






# done decoder

# text proj




# Note that the input checkpoint is the original decoder checkpoint


# done text proj

# super res unet first steps

SUPER_RES_UNET_FIRST_STEPS_PREFIX = "model_first_steps"

SUPER_RES_UNET_FIRST_STEPS_CONFIG = {
    "sample_size": 256,
    "layers_per_block": 3,
    "down_block_types": (
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
    ),
    "up_block_types": (
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
    ),
    "block_out_channels": (320, 640, 960, 1280),
    "in_channels": 6,
    "out_channels": 3,
    "add_attention": False,
}






# done super res unet first steps

# super res unet last step

SUPER_RES_UNET_LAST_STEP_PREFIX = "model_last_step"

SUPER_RES_UNET_LAST_STEP_CONFIG = {
    "sample_size": 256,
    "layers_per_block": 3,
    "down_block_types": (
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
    ),
    "up_block_types": (
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
    ),
    "block_out_channels": (320, 640, 960, 1280),
    "in_channels": 6,
    "out_channels": 3,
    "add_attention": False,
}






# done super res unet last step


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


# Driver functions












if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    parser.add_argument(
        "--prior_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the prior checkpoint to convert.",
    )

    parser.add_argument(
        "--decoder_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the decoder checkpoint to convert.",
    )

    parser.add_argument(
        "--super_res_unet_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the super resolution checkpoint to convert.",
    )

    parser.add_argument(
        "--clip_stat_path", default=None, type=str, required=True, help="Path to the clip stats checkpoint to convert."
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
        text_encoder_model, tokenizer_model = text_encoder()

        prior_model = prior(args=args, checkpoint_map_location=checkpoint_map_location)

        decoder_model, text_proj_model = decoder(args=args, checkpoint_map_location=checkpoint_map_location)

        super_res_first_model, super_res_last_model = super_res_unet(
            args=args, checkpoint_map_location=checkpoint_map_location
        )

        prior_scheduler = UnCLIPScheduler(
            variance_type="fixed_small_log",
            prediction_type="sample",
            num_train_timesteps=1000,
            clip_sample_range=5.0,
        )

        decoder_scheduler = UnCLIPScheduler(
            variance_type="learned_range",
            prediction_type="epsilon",
            num_train_timesteps=1000,
        )

        super_res_scheduler = UnCLIPScheduler(
            variance_type="fixed_small_log",
            prediction_type="epsilon",
            num_train_timesteps=1000,
        )

        print(f"saving Kakao Brain unCLIP to {args.dump_path}")

        pipe = UnCLIPPipeline(
            prior=prior_model,
            decoder=decoder_model,
            text_proj=text_proj_model,
            tokenizer=tokenizer_model,
            text_encoder=text_encoder_model,
            super_res_first=super_res_first_model,
            super_res_last=super_res_last_model,
            prior_scheduler=prior_scheduler,
            decoder_scheduler=decoder_scheduler,
            super_res_scheduler=super_res_scheduler,
        )
        pipe.save_pretrained(args.dump_path)

        print("done writing Kakao Brain unCLIP")
    elif args.debug == "text_encoder":
        text_encoder_model, tokenizer_model = text_encoder()
    elif args.debug == "prior":
        prior_model = prior(args=args, checkpoint_map_location=checkpoint_map_location)
    elif args.debug == "decoder":
        decoder_model, text_proj_model = decoder(args=args, checkpoint_map_location=checkpoint_map_location)
    elif args.debug == "super_res_unet":
        super_res_first_model, super_res_last_model = super_res_unet(
            args=args, checkpoint_map_location=checkpoint_map_location
        )
    else:
        raise ValueError(f"unknown debug value : {args.debug}")