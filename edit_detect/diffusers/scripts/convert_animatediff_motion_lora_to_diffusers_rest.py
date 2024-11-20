import argparse

import torch
from safetensors.torch import load_file, save_file






if __name__ == "__main__":
    args = get_args()

    if args.ckpt_path.endswith(".safetensors"):
        state_dict = load_file(args.ckpt_path)
    else:
        state_dict = torch.load(args.ckpt_path, map_location="cpu")

    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]

    conv_state_dict = convert_motion_module(state_dict)

    # convert to new format
    output_dict = {}
    for module_name, params in conv_state_dict.items():
        if type(params) is not torch.Tensor:
            continue
        output_dict.update({f"unet.{module_name}": params})

    save_file(output_dict, f"{args.output_path}/diffusion_pytorch_model.safetensors")