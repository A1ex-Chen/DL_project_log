import argparse

import torch
import yaml

from diffusers import DDIMScheduler, LDMPipeline, UNetLDMModel, VQModel




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    convert_ldm_original(args.checkpoint_path, args.config_path, args.output_path)