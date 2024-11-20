import argparse
import os

import torch

from diffusers import (
    CMStochasticIterativeScheduler,
    ConsistencyModelPipeline,
    UNet2DModel,
)


TEST_UNET_CONFIG = {
    "sample_size": 32,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 2,
    "num_class_embeds": 1000,
    "block_out_channels": [32, 64],
    "attention_head_dim": 8,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "AttnDownBlock2D",
    ],
    "up_block_types": [
        "AttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "resnet_time_scale_shift": "scale_shift",
    "attn_norm_num_groups": 32,
    "upsample_type": "resnet",
    "downsample_type": "resnet",
}

IMAGENET_64_UNET_CONFIG = {
    "sample_size": 64,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 3,
    "num_class_embeds": 1000,
    "block_out_channels": [192, 192 * 2, 192 * 3, 192 * 4],
    "attention_head_dim": 64,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ],
    "up_block_types": [
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "resnet_time_scale_shift": "scale_shift",
    "attn_norm_num_groups": 32,
    "upsample_type": "resnet",
    "downsample_type": "resnet",
}

LSUN_256_UNET_CONFIG = {
    "sample_size": 256,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 2,
    "num_class_embeds": None,
    "block_out_channels": [256, 256, 256 * 2, 256 * 2, 256 * 4, 256 * 4],
    "attention_head_dim": 64,
    "down_block_types": [
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "ResnetDownsampleBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ],
    "up_block_types": [
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
        "ResnetUpsampleBlock2D",
    ],
    "resnet_time_scale_shift": "default",
    "upsample_type": "resnet",
    "downsample_type": "resnet",
}

CD_SCHEDULER_CONFIG = {
    "num_train_timesteps": 40,
    "sigma_min": 0.002,
    "sigma_max": 80.0,
}

CT_IMAGENET_64_SCHEDULER_CONFIG = {
    "num_train_timesteps": 201,
    "sigma_min": 0.002,
    "sigma_max": 80.0,
}

CT_LSUN_256_SCHEDULER_CONFIG = {
    "num_train_timesteps": 151,
    "sigma_min": 0.002,
    "sigma_max": 80.0,
}










if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--unet_path", default=None, type=str, required=True, help="Path to the unet.pt to convert.")
    parser.add_argument(
        "--dump_path", default=None, type=str, required=True, help="Path to output the converted UNet model."
    )
    parser.add_argument("--class_cond", default=True, type=str, help="Whether the model is class-conditional.")

    args = parser.parse_args()
    args.class_cond = str2bool(args.class_cond)

    ckpt_name = os.path.basename(args.unet_path)
    print(f"Checkpoint: {ckpt_name}")

    # Get U-Net config
    if "imagenet64" in ckpt_name:
        unet_config = IMAGENET_64_UNET_CONFIG
    elif "256" in ckpt_name and (("bedroom" in ckpt_name) or ("cat" in ckpt_name)):
        unet_config = LSUN_256_UNET_CONFIG
    elif "test" in ckpt_name:
        unet_config = TEST_UNET_CONFIG
    else:
        raise ValueError(f"Checkpoint type {ckpt_name} is not currently supported.")

    if not args.class_cond:
        unet_config["num_class_embeds"] = None

    converted_unet_ckpt = con_pt_to_diffuser(args.unet_path, unet_config)

    image_unet = UNet2DModel(**unet_config)
    image_unet.load_state_dict(converted_unet_ckpt)

    # Get scheduler config
    if "cd" in ckpt_name or "test" in ckpt_name:
        scheduler_config = CD_SCHEDULER_CONFIG
    elif "ct" in ckpt_name and "imagenet64" in ckpt_name:
        scheduler_config = CT_IMAGENET_64_SCHEDULER_CONFIG
    elif "ct" in ckpt_name and "256" in ckpt_name and (("bedroom" in ckpt_name) or ("cat" in ckpt_name)):
        scheduler_config = CT_LSUN_256_SCHEDULER_CONFIG
    else:
        raise ValueError(f"Checkpoint type {ckpt_name} is not currently supported.")

    cm_scheduler = CMStochasticIterativeScheduler(**scheduler_config)

    consistency_model = ConsistencyModelPipeline(unet=image_unet, scheduler=cm_scheduler)
    consistency_model.save_pretrained(args.dump_path)