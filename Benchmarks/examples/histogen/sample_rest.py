import argparse
import os

import torch
from pixelsnail import PixelSNAIL
from torchvision.utils import save_image
from tqdm import tqdm
from vqvae import VQVAE


@torch.no_grad()




if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--vqvae", type=str)
    parser.add_argument("--top", type=str)
    parser.add_argument("--bottom", type=str)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("filename", type=str)

    args = parser.parse_args()

    model_vqvae = load_model("vqvae", args.vqvae, device)
    model_top = load_model("pixelsnail_top", args.top, device)
    model_bottom = load_model("pixelsnail_bottom", args.bottom, device)

    top_sample = sample_model(model_top, device, args.batch, [32, 32], args.temp)
    bottom_sample = sample_model(
        model_bottom, device, args.batch, [64, 64], args.temp, condition=top_sample
    )

    decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
    decoded_sample = decoded_sample.clamp(-1, 1)

    save_image(decoded_sample, args.filename, normalize=True, range=(-1, 1))