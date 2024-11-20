import argparse
import os

import torch
from torchvision.datasets.utils import download_url

from diffusers import AutoencoderKL, DDIMScheduler, DiTPipeline, Transformer2DModel


pretrained_models = {512: "DiT-XL-2-512x512.pt", 256: "DiT-XL-2-256x256.pt"}






if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_size",
        default=256,
        type=int,
        required=False,
        help="Image size of pretrained model, either 256 or 512.",
    )
    parser.add_argument(
        "--vae_model",
        default="stabilityai/sd-vae-ft-ema",
        type=str,
        required=False,
        help="Path to pretrained VAE model, either stabilityai/sd-vae-ft-mse or stabilityai/sd-vae-ft-ema.",
    )
    parser.add_argument(
        "--save", default=True, type=bool, required=False, help="Whether to save the converted pipeline or not."
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the output pipeline."
    )

    args = parser.parse_args()
    main(args)