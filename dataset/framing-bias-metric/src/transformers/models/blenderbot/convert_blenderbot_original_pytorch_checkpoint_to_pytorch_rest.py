# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert Blenderbot checkpoint."""

import argparse

import torch

from transformers import BartConfig, BartForConditionalGeneration
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

PATTERNS = [
    ["attention", "attn"],
    ["encoder_attention", "encoder_attn"],
    ["q_lin", "q_proj"],
    ["k_lin", "k_proj"],
    ["v_lin", "v_proj"],
    ["out_lin", "out_proj"],
    ["norm_embeddings", "layernorm_embedding"],
    ["position_embeddings", "embed_positions"],
    ["embeddings", "embed_tokens"],
    ["ffn.lin", "fc"],
]






IGNORE_KEYS = ["START"]


@torch.no_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--src_path", type=str, help="like blenderbot-model.bin")
    parser.add_argument("--save_dir", default="hf_blenderbot", type=str, help="Where to save converted model.")
    parser.add_argument(
        "--hf_config_json", default="blenderbot-3b-config.json", type=str, help="Path to config to use"
    )
    args = parser.parse_args()
    convert_parlai_checkpoint(args.src_path, args.save_dir, args.hf_config_json)