import argparse
import csv
import os
import random
import clip
import numpy as np
import torch

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from tqdm import tqdm
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
)


clip_model = "ViT-L/14"

sd_model_dir = "./" #
sd_model_base = f"{sd_model_dir}/stable-diffusion-xl-base-1.0"
sd_model_refiner = f"{sd_model_dir}/stable-diffusion-xl-refiner-1.0"

sd_model_edit = f"{sd_model_dir}/timbrooks/instruct-pix2pix"

exp_dir = "../exp_flip" #  your target generation directory
fail_exp_dir = "../exp_fail" # your target generation directory
sub_exp = "xl_generate_base"
sub_cf_exp = "xl_generate_cf_via_instructpix2pix"

path = "../resources/occ_us.csv"
prompt_path = "../resources/prompts.txt"


class Generator:










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="log.txt",
    )
    args = parser.parse_args()
    Generator(args)