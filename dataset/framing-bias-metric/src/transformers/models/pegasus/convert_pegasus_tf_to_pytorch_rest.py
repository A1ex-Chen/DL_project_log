# coding=utf-8
# Copyright 2020 Google and The HuggingFace Inc. team.
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

import argparse
import os
from pathlib import Path
from typing import Dict

import tensorflow as tf
import torch
from tqdm import tqdm

from transformers import PegasusConfig, PegasusForConditionalGeneration, PegasusTokenizer
from transformers.models.pegasus.configuration_pegasus import DEFAULTS, task_specific_params


PATTERNS = [
    # replace left string with right string to get the relevant state_dict key (identical state dict to bart)
    ["memory_attention", "encoder_attn"],
    ["attention", "attn"],
    ["/", "."],
    [".LayerNorm.gamma", "_layer_norm.weight"],
    [".LayerNorm.beta", "_layer_norm.bias"],
    ["r.layer_", "r.layers."],
    ["output_proj", "out_proj"],
    ["ffn.dense_1.", "fc2."],
    ["ffn.dense.", "fc1."],
    ["ffn_layer_norm", "final_layer_norm"],
    ["kernel", "weight"],
    ["encoder_layer_norm.", "encoder.layer_norm."],
    ["decoder_layer_norm.", "decoder.layer_norm."],
    ["embeddings.weights", "shared.weight"],
]




# See appendix C of paper for all hyperparams








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("tf_ckpt_path", type=str, help="passed to tf.train.list_variables")
    parser.add_argument("save_dir", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    if args.save_dir is None:
        dataset = Path(args.tf_ckpt_path).parent.name
        args.save_dir = os.path.join("pegasus", dataset)
    convert_pegasus_ckpt_to_pytorch(args.tf_ckpt_path, args.save_dir)