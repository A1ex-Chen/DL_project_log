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
"""Conversion script for the NCSNPP checkpoints."""

import argparse
import json

import torch

from diffusers import ScoreSdeVePipeline, ScoreSdeVeScheduler, UNet2DModel




    for i, block in enumerate(new_model_architecture.downsample_blocks):
        has_attentions = hasattr(block, "attentions")
        for j in range(len(block.resnets)):
            set_resnet_weights(block.resnets[j], checkpoint, module_index)
            module_index += 1
            if has_attentions:
                set_attention_weights(block.attentions[j], checkpoint, module_index)
                module_index += 1

        if hasattr(block, "downsamplers") and block.downsamplers is not None:
            set_resnet_weights(block.resnet_down, checkpoint, module_index)
            module_index += 1
            block.skip_conv.weight.data = checkpoint[f"all_modules.{module_index}.Conv_0.weight"].data
            block.skip_conv.bias.data = checkpoint[f"all_modules.{module_index}.Conv_0.bias"].data
            module_index += 1

    set_resnet_weights(new_model_architecture.mid_block.resnets[0], checkpoint, module_index)
    module_index += 1
    set_attention_weights(new_model_architecture.mid_block.attentions[0], checkpoint, module_index)
    module_index += 1
    set_resnet_weights(new_model_architecture.mid_block.resnets[1], checkpoint, module_index)
    module_index += 1

    for i, block in enumerate(new_model_architecture.up_blocks):
        has_attentions = hasattr(block, "attentions")
        for j in range(len(block.resnets)):
            set_resnet_weights(block.resnets[j], checkpoint, module_index)
            module_index += 1
        if has_attentions:
            set_attention_weights(
                block.attentions[0], checkpoint, module_index
            )  # why can there only be a single attention layer for up?
            module_index += 1

        if hasattr(block, "resnet_up") and block.resnet_up is not None:
            block.skip_norm.weight.data = checkpoint[f"all_modules.{module_index}.weight"].data
            block.skip_norm.bias.data = checkpoint[f"all_modules.{module_index}.bias"].data
            module_index += 1
            block.skip_conv.weight.data = checkpoint[f"all_modules.{module_index}.weight"].data
            block.skip_conv.bias.data = checkpoint[f"all_modules.{module_index}.bias"].data
            module_index += 1
            set_resnet_weights(block.resnet_up, checkpoint, module_index)
            module_index += 1

    new_model_architecture.conv_norm_out.weight.data = checkpoint[f"all_modules.{module_index}.weight"].data
    new_model_architecture.conv_norm_out.bias.data = checkpoint[f"all_modules.{module_index}.bias"].data
    module_index += 1
    new_model_architecture.conv_out.weight.data = checkpoint[f"all_modules.{module_index}.weight"].data
    new_model_architecture.conv_out.bias.data = checkpoint[f"all_modules.{module_index}.bias"].data

    return new_model_architecture.state_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        default="/Users/arthurzucker/Work/diffusers/ArthurZ/diffusion_pytorch_model.bin",
        type=str,
        required=False,
        help="Path to the checkpoint to convert.",
    )

    parser.add_argument(
        "--config_file",
        default="/Users/arthurzucker/Work/diffusers/ArthurZ/config.json",
        type=str,
        required=False,
        help="The config json file corresponding to the architecture.",
    )

    parser.add_argument(
        "--dump_path",
        default="/Users/arthurzucker/Work/diffusers/ArthurZ/diffusion_model_new.pt",
        type=str,
        required=False,
        help="Path to the output model.",
    )

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

    with open(args.config_file) as f:
        config = json.loads(f.read())

    converted_checkpoint = convert_ncsnpp_checkpoint(
        checkpoint,
        config,
    )

    if "sde" in config:
        del config["sde"]

    model = UNet2DModel(**config)
    model.load_state_dict(converted_checkpoint)

    try:
        scheduler = ScoreSdeVeScheduler.from_config("/".join(args.checkpoint_path.split("/")[:-1]))

        pipe = ScoreSdeVePipeline(unet=model, scheduler=scheduler)
        pipe.save_pretrained(args.dump_path)
    except:  # noqa: E722
        model.save_pretrained(args.dump_path)