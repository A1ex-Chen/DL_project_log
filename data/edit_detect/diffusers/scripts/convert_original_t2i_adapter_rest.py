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
"""
Conversion script for the T2I-Adapter checkpoints.
"""

import argparse

import torch

from diffusers import T2IAdapter






if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--output_path", default=None, type=str, required=True, help="Path to the store the result checkpoint."
    )
    parser.add_argument(
        "--is_adapter_light",
        action="store_true",
        help="Is checkpoint come from Adapter-Light architecture. ex: color-adapter",
    )
    parser.add_argument("--in_channels", required=False, type=int, help="Input channels for non-light adapter")

    args = parser.parse_args()
    src_state = torch.load(args.checkpoint_path)

    if args.is_adapter_light:
        adapter = convert_light_adapter(src_state)
    else:
        if args.in_channels is None:
            raise ValueError("set `--in_channels=<n>`")
        adapter = convert_adapter(src_state, args.in_channels)

    adapter.save_pretrained(args.output_path)