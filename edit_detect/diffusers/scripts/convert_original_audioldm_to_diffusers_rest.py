# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Conversion script for the AudioLDM checkpoints."""

import argparse
import re

import torch
import yaml
from transformers import (
    AutoTokenizer,
    ClapTextConfig,
    ClapTextModelWithProjection,
    SpeechT5HifiGan,
    SpeechT5HifiGanConfig,
)

from diffusers import (
    AudioLDMPipeline,
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.shave_segments


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.renew_resnet_paths


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.renew_vae_resnet_paths


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.renew_attention_paths


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.renew_vae_attention_paths


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.assign_to_checkpoint


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.conv_attn_to_linear




# Adapted from diffusers.pipelines.stable_diffusion.convert_from_ckpt.create_vae_diffusers_config


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.create_diffusers_schedular


# Adapted from diffusers.pipelines.stable_diffusion.convert_from_ckpt.convert_ldm_unet_checkpoint


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.convert_ldm_vae_checkpoint


CLAP_KEYS_TO_MODIFY_MAPPING = {
    "text_branch": "text_model",
    "attn": "attention.self",
    "self.proj": "output.dense",
    "attention.self_mask": "attn_mask",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
    "norm1": "layernorm_before",
    "norm2": "layernorm_after",
    "bn0": "batch_norm",
}

CLAP_KEYS_TO_IGNORE = ["text_transform"]

CLAP_EXPECTED_MISSING_KEYS = ["text_model.embeddings.token_type_ids"]








# Adapted from https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation/blob/84a0384742a22bd80c44e903e241f0623e874f1d/audioldm/utils.py#L72-L73
DEFAULT_CONFIG = {
    "model": {
        "params": {
            "linear_start": 0.0015,
            "linear_end": 0.0195,
            "timesteps": 1000,
            "channels": 8,
            "scale_by_std": True,
            "unet_config": {
                "target": "audioldm.latent_diffusion.openaimodel.UNetModel",
                "params": {
                    "extra_film_condition_dim": 512,
                    "extra_film_use_concat": True,
                    "in_channels": 8,
                    "out_channels": 8,
                    "model_channels": 128,
                    "attention_resolutions": [8, 4, 2],
                    "num_res_blocks": 2,
                    "channel_mult": [1, 2, 3, 5],
                    "num_head_channels": 32,
                },
            },
            "first_stage_config": {
                "target": "audioldm.variational_autoencoder.autoencoder.AutoencoderKL",
                "params": {
                    "embed_dim": 8,
                    "ddconfig": {
                        "z_channels": 8,
                        "resolution": 256,
                        "in_channels": 1,
                        "out_ch": 1,
                        "ch": 128,
                        "ch_mult": [1, 2, 4],
                        "num_res_blocks": 2,
                    },
                },
            },
            "vocoder_config": {
                "target": "audioldm.first_stage_model.vocoder",
                "params": {
                    "upsample_rates": [5, 4, 2, 2, 2],
                    "upsample_kernel_sizes": [16, 16, 8, 4, 4],
                    "upsample_initial_channel": 1024,
                    "resblock_kernel_sizes": [3, 7, 11],
                    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    "num_mels": 64,
                    "sampling_rate": 16000,
                },
            },
        },
    },
}




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
        "--num_in_channels",
        default=None,
        type=int,
        help="The number of input channels. If `None` number of input channels will be automatically inferred.",
    )
    parser.add_argument(
        "--model_channels",
        default=None,
        type=int,
        help="The number of UNet model channels. If `None`, it will be automatically inferred from the config. Override"
        " to 128 for the small checkpoints, 192 for the medium checkpoints and 256 for the large.",
    )
    parser.add_argument(
        "--num_head_channels",
        default=None,
        type=int,
        help="The number of UNet head channels. If `None`, it will be automatically inferred from the config. Override"
        " to 32 for the small and medium checkpoints, and 64 for the large.",
    )
    parser.add_argument(
        "--scheduler_type",
        default="ddim",
        type=str,
        help="Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']",
    )
    parser.add_argument(
        "--image_size",
        default=None,
        type=int,
        help=("The image size that the model was trained on."),
    )
    parser.add_argument(
        "--prediction_type",
        default=None,
        type=str,
        help=("The prediction type that the model was trained on."),
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
        "--from_safetensors",
        action="store_true",
        help="If `--checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.",
    )
    parser.add_argument(
        "--to_safetensors",
        action="store_true",
        help="Whether to store pipeline in safetensors format or not.",
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--device", type=str, help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    args = parser.parse_args()

    pipe = load_pipeline_from_original_audioldm_ckpt(
        checkpoint_path=args.checkpoint_path,
        original_config_file=args.original_config_file,
        image_size=args.image_size,
        prediction_type=args.prediction_type,
        extract_ema=args.extract_ema,
        scheduler_type=args.scheduler_type,
        num_in_channels=args.num_in_channels,
        model_channels=args.model_channels,
        num_head_channels=args.num_head_channels,
        from_safetensors=args.from_safetensors,
        device=args.device,
    )
    pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)