"""
This script modified from
https://github.com/huggingface/diffusers/blob/bc691231360a4cbc7d19a58742ebb8ed0f05e027/scripts/convert_original_stable_diffusion_to_diffusers.py

Convert original Zero1to3 checkpoint to diffusers checkpoint.

# run the convert script
$ python convert_zero123_to_diffusers.py \
   --checkpoint_path /path/zero123/105000.ckpt \
   --dump_path ./zero1to3 \
   --original_config_file /path/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml
```
"""

import argparse

import torch
import yaml
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from pipeline_zero1to3 import CCProjection, Zero1to3StableDiffusionPipeline
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

from diffusers.models import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import logging


logger = logging.get_logger(__name__)


























if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        help="The YAML config file corresponding to the original architecture.",
    )
    parser.add_argument(
        "--extract_ema",
        action="store_true",
        help=(
            "Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights"
            " or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield"
            " higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning."
        ),
    )
    parser.add_argument(
        "--to_safetensors",
        action="store_true",
        help="Whether to store pipeline in safetensors format or not.",
    )
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--device", type=str, help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    args = parser.parse_args()

    pipe = convert_from_original_zero123_ckpt(
        checkpoint_path=args.checkpoint_path,
        original_config_file=args.original_config_file,
        extract_ema=args.extract_ema,
        device=args.device,
    )

    if args.half:
        pipe.to(dtype=torch.float16)

    pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)