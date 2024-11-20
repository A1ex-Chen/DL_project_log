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
"""Conversion script for the AudioLDM2 checkpoints."""

import argparse
import re
from typing import List, Union

import torch
import yaml
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    ClapConfig,
    ClapModel,
    GPT2Config,
    GPT2Model,
    SpeechT5HifiGan,
    SpeechT5HifiGanConfig,
    T5Config,
    T5EncoderModel,
)

from diffusers import (
    AudioLDM2Pipeline,
    AudioLDM2ProjectionModel,
    AudioLDM2UNet2DConditionModel,
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import is_safetensors_available
from diffusers.utils.import_utils import BACKENDS_MAPPING


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.shave_segments


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.renew_resnet_paths


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.renew_vae_resnet_paths


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.renew_attention_paths










# Adapted from diffusers.pipelines.stable_diffusion.convert_from_ckpt.create_vae_diffusers_config


# Copied from diffusers.pipelines.stable_diffusion.convert_from_ckpt.create_diffusers_schedular






CLAP_KEYS_TO_MODIFY_MAPPING = {
    "text_branch": "text_model",
    "audio_branch": "audio_model.audio_encoder",
    "attn": "attention.self",
    "self.proj": "output.dense",
    "attention.self_mask": "attn_mask",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
    "norm1": "layernorm_before",
    "norm2": "layernorm_after",
    "bn0": "batch_norm",
}

CLAP_KEYS_TO_IGNORE = [
    "text_transform",
    "audio_transform",
    "stft",
    "logmel_extractor",
    "tscam_conv",
    "head",
    "attn_mask",
]

CLAP_EXPECTED_MISSING_KEYS = ["text_model.embeddings.token_type_ids"]












# Adapted from https://github.com/haoheliu/AudioLDM2/blob/81ad2c6ce015c1310387695e2dae975a7d2ed6fd/audioldm2/utils.py#L143
DEFAULT_CONFIG = {
    "model": {
        "params": {
            "linear_start": 0.0015,
            "linear_end": 0.0195,
            "timesteps": 1000,
            "channels": 8,
            "scale_by_std": True,
            "unet_config": {
                "target": "audioldm2.latent_diffusion.openaimodel.UNetModel",
                "params": {
                    "context_dim": [None, 768, 1024],
                    "in_channels": 8,
                    "out_channels": 8,
                    "model_channels": 128,
                    "attention_resolutions": [8, 4, 2],
                    "num_res_blocks": 2,
                    "channel_mult": [1, 2, 3, 5],
                    "num_head_channels": 32,
                    "transformer_depth": 1,
                },
            },
            "first_stage_config": {
                "target": "audioldm2.variational_autoencoder.autoencoder.AutoencoderKL",
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
            "cond_stage_config": {
                "crossattn_audiomae_generated": {
                    "target": "audioldm2.latent_diffusion.modules.encoders.modules.SequenceGenAudioMAECond",
                    "params": {
                        "sequence_gen_length": 8,
                        "sequence_input_embed_dim": [512, 1024],
                    },
                }
            },
            "vocoder_config": {
                "target": "audioldm2.first_stage_model.vocoder",
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
        "--cross_attention_dim",
        default=None,
        type=int,
        nargs="+",
        help="The dimension of the cross-attention layers. If `None`, the cross-attention dimension will be "
        "automatically inferred. Set to `768+1024` for the base model, or `768+1024+640` for the large model",
    )
    parser.add_argument(
        "--transformer_layers_per_block",
        default=None,
        type=int,
        help="The number of transformer layers in each transformer block. If `None`, number of layers will be "
        "automatically inferred. Set to `1` for the base model, or `2` for the large model.",
    )
    parser.add_argument(
        "--scheduler_type",
        default="ddim",
        type=str,
        help="Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']",
    )
    parser.add_argument(
        "--image_size",
        default=1048,
        type=int,
        help="The image size that the model was trained on.",
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

    pipe = load_pipeline_from_original_AudioLDM2_ckpt(
        checkpoint_path=args.checkpoint_path,
        original_config_file=args.original_config_file,
        image_size=args.image_size,
        prediction_type=args.prediction_type,
        extract_ema=args.extract_ema,
        scheduler_type=args.scheduler_type,
        cross_attention_dim=args.cross_attention_dim,
        transformer_layers_per_block=args.transformer_layers_per_block,
        from_safetensors=args.from_safetensors,
        device=args.device,
    )
    pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)