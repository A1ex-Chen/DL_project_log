import argparse
import os

import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, PixArtSigmaPipeline, Transformer2DModel


ckpt_id = "PixArt-alpha"
# https://github.com/PixArt-alpha/PixArt-sigma/blob/dd087141864e30ec44f12cb7448dd654be065e88/scripts/inference.py#L158
interpolation_scale = {256: 0.5, 512: 1, 1024: 2, 2048: 4}




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--micro_condition", action="store_true", help="If use Micro-condition in PixArtMS structure during training."
    )
    parser.add_argument("--qk_norm", action="store_true", help="If use qk norm during training.")
    parser.add_argument(
        "--orig_ckpt_path", default=None, type=str, required=False, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--image_size",
        default=1024,
        type=int,
        choices=[256, 512, 1024, 2048],
        required=False,
        help="Image size of pretrained model, 256, 512, 1024, or 2048.",
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output pipeline.")
    parser.add_argument("--only_transformer", default=True, type=bool, required=True)

    args = parser.parse_args()
    main(args)