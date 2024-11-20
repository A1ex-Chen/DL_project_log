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
import json

import torch

from diffusers import DDPMScheduler, LDMPipeline, UNet2DModel, VQModel












if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )

    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the architecture.",
    )

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path)

    with open(args.config_file) as f:
        config = json.loads(f.read())

    converted_checkpoint = convert_ldm_checkpoint(checkpoint, config)

    if "ldm" in config:
        del config["ldm"]

    model = UNet2DModel(**config)
    model.load_state_dict(converted_checkpoint)

    try:
        scheduler = DDPMScheduler.from_config("/".join(args.checkpoint_path.split("/")[:-1]))
        vqvae = VQModel.from_pretrained("/".join(args.checkpoint_path.split("/")[:-1]))

        pipe = LDMPipeline(unet=model, scheduler=scheduler, vae=vqvae)
        pipe.save_pretrained(args.dump_path)
    except:  # noqa: E722
        model.save_pretrained(args.dump_path)