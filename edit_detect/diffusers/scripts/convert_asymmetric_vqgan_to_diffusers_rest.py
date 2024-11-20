import argparse
import time
from pathlib import Path
from typing import Any, Dict, Literal

import torch

from diffusers import AsymmetricAutoencoderKL


ASYMMETRIC_AUTOENCODER_KL_x_1_5_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ],
    "down_block_out_channels": [128, 256, 512, 512],
    "layers_per_down_block": 2,
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
    "up_block_out_channels": [192, 384, 768, 768],
    "layers_per_up_block": 3,
    "act_fn": "silu",
    "latent_channels": 4,
    "norm_num_groups": 32,
    "sample_size": 256,
    "scaling_factor": 0.18215,
}

ASYMMETRIC_AUTOENCODER_KL_x_2_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ],
    "down_block_out_channels": [128, 256, 512, 512],
    "layers_per_down_block": 2,
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
    "up_block_out_channels": [256, 512, 1024, 1024],
    "layers_per_up_block": 5,
    "act_fn": "silu",
    "latent_channels": 4,
    "norm_num_groups": 32,
    "sample_size": 256,
    "scaling_factor": 0.18215,
}






if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        default=None,
        type=str,
        required=True,
        help="Asymmetric VQGAN scale: `1.5` or `2`",
    )
    parser.add_argument(
        "--original_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the original Asymmetric VQGAN checkpoint",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="Path to save pretrained AsymmetricAutoencoderKL model",
    )
    parser.add_argument(
        "--map_location",
        default="cpu",
        type=str,
        required=False,
        help="The device passed to `map_location` when loading the checkpoint",
    )
    args = parser.parse_args()

    assert args.scale in ["1.5", "2"], f"{args.scale} should be `1.5` of `2`"
    assert Path(args.original_checkpoint_path).is_file()

    asymmetric_autoencoder_kl = get_asymmetric_autoencoder_kl_from_original_checkpoint(
        scale=args.scale,
        original_checkpoint_path=args.original_checkpoint_path,
        map_location=torch.device(args.map_location),
    )
    print("Saving pretrained AsymmetricAutoencoderKL")
    asymmetric_autoencoder_kl.save_pretrained(args.output_path)
    print(f"Done in {time.time() - start:.2f} seconds")