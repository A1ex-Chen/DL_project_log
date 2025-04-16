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
"""Conversion script for the LDM checkpoints."""

import argparse

import torch

from diffusers import UNet3DConditionModel














if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    args = parser.parse_args()

    unet_checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    unet = UNet3DConditionModel()

    converted_ckpt = convert_ldm_unet_checkpoint(unet_checkpoint, unet.config)

    diff_0 = set(unet.state_dict().keys()) - set(converted_ckpt.keys())
    diff_1 = set(converted_ckpt.keys()) - set(unet.state_dict().keys())

    assert len(diff_0) == len(diff_1) == 0, "Converted weights don't match"

    # load state_dict
    unet.load_state_dict(converted_ckpt)

    unet.save_pretrained(args.dump_path)

    # -- finish converting the unet --