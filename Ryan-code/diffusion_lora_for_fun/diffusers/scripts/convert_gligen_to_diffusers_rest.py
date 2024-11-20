import argparse
import re

import torch
import yaml
from transformers import (
    CLIPProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionGLIGENPipeline,
    StableDiffusionGLIGENTextImagePipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    assign_to_checkpoint,
    conv_attn_to_linear,
    protected,
    renew_attention_paths,
    renew_resnet_paths,
    renew_vae_attention_paths,
    renew_vae_resnet_paths,
    shave_segments,
    textenc_conversion_map,
    textenc_pattern,
)














if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        required=True,
        help="The YAML config file corresponding to the gligen architecture.",
    )
    parser.add_argument(
        "--num_in_channels",
        default=None,
        type=int,
        help="The number of input channels. If `None` number of input channels will be automatically inferred.",
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
        "--attention_type",
        default=None,
        type=str,
        required=True,
        help="Type of attention ex: gated or gated-text-image",
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--device", type=str, help="Device to use.")
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")

    args = parser.parse_args()

    pipe = convert_gligen_to_diffusers(
        checkpoint_path=args.checkpoint_path,
        original_config_file=args.original_config_file,
        attention_type=args.attention_type,
        extract_ema=args.extract_ema,
        num_in_channels=args.num_in_channels,
        device=args.device,
    )

    if args.half:
        pipe.to(dtype=torch.float16)

    pipe.save_pretrained(args.dump_path)