"""
This script requires you to build `LAVIS` from source, since the pip version doesn't have BLIP Diffusion. Follow instructions here: https://github.com/salesforce/LAVIS/tree/main.
"""

import argparse
import os
import tempfile

import torch
from lavis.models import load_model_and_preprocess
from transformers import CLIPTokenizer
from transformers.models.blip_2.configuration_blip_2 import Blip2Config

from diffusers import (
    AutoencoderKL,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines import BlipDiffusionPipeline
from diffusers.pipelines.blip_diffusion.blip_image_processing import BlipImageProcessor
from diffusers.pipelines.blip_diffusion.modeling_blip2 import Blip2QFormerModel
from diffusers.pipelines.blip_diffusion.modeling_ctx_clip import ContextCLIPTextModel


BLIP2_CONFIG = {
    "vision_config": {
        "hidden_size": 1024,
        "num_hidden_layers": 23,
        "num_attention_heads": 16,
        "image_size": 224,
        "patch_size": 14,
        "intermediate_size": 4096,
        "hidden_act": "quick_gelu",
    },
    "qformer_config": {
        "cross_attention_frequency": 1,
        "encoder_hidden_size": 1024,
        "vocab_size": 30523,
    },
    "num_query_tokens": 16,
}
blip2config = Blip2Config(**BLIP2_CONFIG)




























if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Path to the output model.")
    args = parser.parse_args()

    main(args)