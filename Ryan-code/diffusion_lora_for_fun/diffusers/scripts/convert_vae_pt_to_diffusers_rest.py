import argparse
import io

import requests
import torch
import yaml

from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    assign_to_checkpoint,
    conv_attn_to_linear,
    create_vae_diffusers_config,
    renew_vae_attention_paths,
    renew_vae_resnet_paths,
)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--vae_pt_path", default=None, type=str, required=True, help="Path to the VAE.pt to convert.")
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the VAE.pt to convert.")

    args = parser.parse_args()

    vae_pt_to_vae_diffuser(args.vae_pt_path, args.dump_path)