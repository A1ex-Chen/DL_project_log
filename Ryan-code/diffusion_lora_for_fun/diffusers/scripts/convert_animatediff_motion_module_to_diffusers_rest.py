import argparse

import torch
from safetensors.torch import load_file

from diffusers import MotionAdapter






if __name__ == "__main__":
    args = get_args()

    if args.ckpt_path.endswith(".safetensors"):
        state_dict = load_file(args.ckpt_path)
    else:
        state_dict = torch.load(args.ckpt_path, map_location="cpu")

    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]

    conv_state_dict = convert_motion_module(state_dict)
    adapter = MotionAdapter(
        block_out_channels=args.block_out_channels,
        use_motion_mid_block=args.use_motion_mid_block,
        motion_max_seq_length=args.motion_max_seq_length,
    )
    # skip loading position embeddings
    adapter.load_state_dict(conv_state_dict, strict=False)
    adapter.save_pretrained(args.output_path)

    if args.save_fp16:
        adapter.to(dtype=torch.float16).save_pretrained(args.output_path, variant="fp16")